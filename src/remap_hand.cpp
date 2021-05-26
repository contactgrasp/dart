//
// Created by samarth on 6/25/18.
//

#define CUDA_ERR_CHECK

#include <iostream>

// OpenGL
#include <GL/glew.h>

// Pangolin
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

// CUDA

// DART
#include "remap_hand.hpp"
#include "util/dart_io.h"

using namespace std;
namespace fs = boost::filesystem;


HandMapper::HandMapper(const std::string &src_model_filename,
    const std::string &dst_model_filename, bool graspit,
    const std::string &data_dir) :
    src_hand_model_filename(src_model_filename),
    dst_hand_model_filename(dst_model_filename), data_dir(data_dir),
    graspit(graspit),
    _iterate(false), _init(false), _show_axes(false), _calc_error(false),
    _save_result(false),
    _show_priors(false), _show_object(true),
    _show_src_hand(true), _show_dst_hand(false),
    _reset_joints(false),
    _max_iterations(160),
    _attract_weight(150.f),
    _inter_model_intersection(0.f), _intra_model_intersection(5.f),
    _lm_damping(25.f), _log10_regularization(-1.f),
    src_hand_id(0), dst_hand_id(1), n_priors(50),
    tracker_ready(false), analyze_from_pango_thread(false),
    quit_pangolin_loop(false)
{
  // start the pangolin thread
  pango_thread = make_shared<thread>(&HandMapper::pangolin_loop, this);

  // wait for tracker to be ready
  if (!tracker_ready) {
    unique_lock<mutex> lk(tracker_mutex);
    tracker_cv.wait(lk, [this]{return this->tracker_ready;});
  }

  T_s_d = dart::SE3FromTranslation(-0.14, 0, 0) *
      dart::SE3FromRotationY(-M_PI/2) * dart::SE3FromRotationZ(-M_PI/2);
}


void HandMapper::create_tracker() {
  tracker = make_shared<dart::TrackerNoObs>();
  tracker->addModel(src_hand_model_filename.string());
  tracker->addModel(dst_hand_model_filename.string());
  fs::path self_intersection_filename = dst_hand_model_filename.parent_path() /
    "intersection_potential_matrix.txt";
  int *hand_self_intersection = dart::loadSelfIntersectionMatrix(
      self_intersection_filename.string(),
      tracker->getModel(dst_hand_id).getNumSdfs());
  tracker->setIntersectionPotentialMatrix(dst_hand_id, hand_self_intersection);
  delete [] hand_self_intersection;  // create sliders

  // create slider vars
  auto &pose  = tracker->getPose(src_hand_id);
  for (int i=0; i < pose.getReducedArticulatedDimensions(); i++) {
    stringstream name;
    name << "ui.dof" << i;
    pangolin::Var<float> slider(name.str(), 0.f,
        pose.getReducedMin(i), pose.getReducedMax(i));
    sliders.push_back(slider);
  }

  // mapping from graspit DOFs to dart DOFs
  graspit2dartdofs.resize(
      tracker->getPose(dst_hand_id).getReducedArticulatedDimensions());
  iota(graspit2dartdofs.begin(), graspit2dartdofs.end(), 0);

  // zero out the hand-object intersection matrix
  int n = tracker->getNumModels();
  memset(tracker->getOptions().lambdaIntersection.data(), 0, n*n*sizeof(float));
}


void HandMapper::save_result(const std::string &src_filename) {
  stringstream f;
  f << "-1,";
  const auto &pose = tracker->getPose(dst_hand_id);
  auto const &palm_pose = pose.getTransformModelToCamera();
  f << palm_pose.r0.x << ",";
  f << palm_pose.r0.y << ",";
  f << palm_pose.r0.z << ",";
  f << palm_pose.r0.w << ",";
  f << palm_pose.r1.x << ",";
  f << palm_pose.r1.y << ",";
  f << palm_pose.r1.z << ",";
  f << palm_pose.r1.w << ",";
  f << palm_pose.r2.x << ",";
  f << palm_pose.r2.y << ",";
  f << palm_pose.r2.z << ",";
  f << palm_pose.r2.w;
  for (int i = 0; i < pose.getReducedArticulatedDimensions(); i++)
    f << "," << pose.getReducedArticulation()[i];

  stringstream dst_filename;
  if (graspit) {
    dst_filename << "graspit_allegro_mapped_";
    dst_filename << src_filename.substr(src_filename.find_first_of('_')+1);
  } else {
    dst_filename << "allegro_mapped_" << src_filename;
  }
  fs::path out_filename = data_dir / dst_filename.str();
  ofstream of(out_filename.string());
  if (!of.is_open()) {
    cout << "Could not open " << out_filename << " for writing" << endl;
    return;
  }
  of << f.str() << endl;
  of.close();
  cout << "Mapped hand pose written to " << out_filename << endl;
  // quit_pangolin_loop = true;
}


void HandMapper::set_src_pose(const std::string &pose_filename) {
  // open file
  fs::path filename = data_dir / pose_filename;
  ifstream f(filename.string());
  if (!f.is_open()) {
    cout << "Could not open " << filename << " for reading" << endl;
    return;
  }

  string line;
  getline(f, line);
  f.close();
  stringstream ss(line);

  // parse the line
  auto &pose = tracker->getPose(src_hand_id);
  vector<float> vals;
  char comma;
  float val;
  ss >> val >> comma;  // absorb the grasp idx
  while (ss >> val) {
    vals.push_back(val);
    ss >> comma;
  }
  if (vals.size() < 12 + pose.getReducedArticulatedDimensions()) {
    cout << "Need " << 12+pose.getReducedArticulatedDimensions() << " dofs in"
        << " file, got only " << vals.size() << endl;
    return;
  }

  size_t dof_idx(0);
  // read palm pose
  vector<float4> palm_pose(3);
  for (int i = 0; i < palm_pose.size(); i++) {
    palm_pose[i].x = vals[dof_idx]; dof_idx++;
    palm_pose[i].y = vals[dof_idx]; dof_idx++;
    palm_pose[i].z = vals[dof_idx]; dof_idx++;
    palm_pose[i].w = vals[dof_idx]; dof_idx++;
  }
  pose.setTransformModelToCamera(dart::SE3(palm_pose[0], palm_pose[1],
      palm_pose[2]));
  // read dofs
  for (int i = 0; i < pose.getReducedArticulatedDimensions(); i++) {
    sliders[i] = vals[dof_idx];
    dof_idx++;
  }
  // allow some time for sliders to take effect through the pango thread
  this_thread::sleep_for(chrono::milliseconds(100));
}


float HandMapper::do_mapping() {
  // map the palm pose
  dart::SE3 src_pose = tracker->getPose(src_hand_id).getTransformModelToCamera();
  dart::SE3 dst_pose = src_pose * T_s_d;
  tracker->getPose(dst_hand_id).setTransformModelToCamera(dst_pose);

  // zero out the dst hand dofs
  for (int i=0; i<tracker->getPose(dst_hand_id).getReducedArticulatedDimensions(); i++) {
    tracker->getPose(dst_hand_id).getReducedArticulation()[i] = 0;
  }
  tracker->updatePose(dst_hand_id);

  // create the 3D-3D priors
  create_priors();

  // optimize
  float error = tracker->optimizePoses();
  cout << "Error = " << error << endl;
  return error;
}


void HandMapper::create_pangolin_vars_callbacks() {
  // setup control buttons and sliders
  pangolin::Var<bool>::Attach("ui.init", _init, false);
  pangolin::Var<bool>::Attach("ui.iterate", _iterate, false);
  pangolin::Var<bool>::Attach("ui.show axes", _show_axes, true);
  pangolin::Var<bool>::Attach("ui.show priors", _show_priors, true);
  pangolin::Var<bool>::Attach("ui.show src hand", _show_src_hand, true);
  pangolin::Var<bool>::Attach("ui.show dst hand", _show_dst_hand, true);
  pangolin::Var<bool>::Attach("ui.show object", _show_object, true);
  pangolin::Var<bool>::Attach("ui.calculate error", _calc_error, false);
  pangolin::Var<int>::Attach("ui.max # iterations", _max_iterations, 0, 200);
  pangolin::Var<float>::Attach("ui.attraction weight", _attract_weight, 0, 400);
  pangolin::Var<float>::Attach("ui.inter model intersection",
      _inter_model_intersection, 0, 100);
  pangolin::Var<float>::Attach("ui.intra model intersection",
      _intra_model_intersection, 0, 100);
  pangolin::Var<float>::Attach("ui.LM damping", _lm_damping, 0, 100);
  pangolin::Var<float>::Attach("ui.log10(regularization)",
      _log10_regularization, -10, 0);

  // callback functions for the sliders
  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandMapper *ga = (HandMapper *)data;
        pangolin::Var<int> v(_var);
        ga->set_max_iterations(v.Get());
      },
      (void *)this, "ui.max # iterations");
  set_max_iterations(_max_iterations);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandMapper *ga = (HandMapper *)data;
        pangolin::Var<float> v(_var);
        ga->set_attract_weight(v.Get());
      },
      (void *)this, "ui.attraction weight");
  set_attract_weight(_attract_weight);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandMapper *ga = (HandMapper *)data;
        pangolin::Var<float> v(_var);
        ga->set_inter_model_intersection(v.Get());
      },
      (void *)this, "ui.inter model intersection");
  set_inter_model_intersection(_inter_model_intersection);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandMapper *ga = (HandMapper *)data;
        pangolin::Var<float> v(_var);
        ga->set_intra_model_intersection(v.Get());
      },
      (void *)this, "ui.intra model intersection");
  set_intra_model_intersection(_intra_model_intersection);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandMapper *ga = (HandMapper *)data;
        pangolin::Var<float> v(_var);
        ga->set_lm_damping(v.Get());
      },
      (void *)this, "ui.LM damping");
  set_lm_damping(_lm_damping);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandMapper *ga = (HandMapper *)data;
        pangolin::Var<float> v(_var);
        ga->set_log10_regularization(v.Get());
      },
      (void *)this, "ui.log10(regularization)");
  set_log10_regularization(_log10_regularization);
}


void HandMapper::pangolin_loop() {
  // OpenGL context needs to be created inside the thread that runs the
  // Pangolin window
  const float im_width(1920), im_height(1200), cam_focal_length(420),
      ui_width(0.3f);
  cudaGLSetGLDevice(0);
  cudaDeviceReset();
  string window_name = "Hand Mapper";
  pangolin::CreateWindowAndBind(window_name, im_width, im_height);

  pangolin::OpenGlRenderState camState(
      pangolin::ProjectionMatrixRDF_TopLeft(im_width, im_height, cam_focal_length,
          cam_focal_length, im_width/2.f, im_height/2.f, 0.01, 1000),
      pangolin::ModelViewLookAtRDF(0, -0.25, 0, 0, 0, 0, 0, 0, 1));
  pangolin::View &camDisp =
      pangolin::Display("cam")
          .SetBounds(0, 1, pangolin::Attach(ui_width), 1, -im_width/im_height)
          .SetHandler(new pangolin::Handler3D(camState));
  // lighting
  glShadeModel(GL_SMOOTH);
  float4 lightPosition =
      make_float4(normalize(make_float3(-0.4405f, -0.5357f, -0.619f)), 0);
  GLfloat light_ambient[] = {0.3, 0.3, 0.3, 1.0};
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_POSITION, (float *)&lightPosition);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);
  glEnable(GL_LIGHTING);
  glColor4ub(0xff, 0xff, 0xff, 0xff);
  glEnable(GL_COLOR_MATERIAL);

  // Variables should be created in the thread that creates the OpenGL context
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach(ui_width));

  // tracker can only be created after OpenGL context has been created
  create_tracker();
  create_pangolin_vars_callbacks();

  // signal that the tracker is ready
  unique_lock<mutex> lk(tracker_mutex);
  tracker_ready = true;
  lk.unlock();
  tracker_cv.notify_one();

  while(!(pangolin::ShouldQuit() || quit_pangolin_loop)) {
    // calculate the error of current pose
    if (pangolin::Pushed(_calc_error))
      cout << "Error = " << tracker->getError() << endl;

    // optimize the poses
    if (pangolin::Pushed(_iterate)) do_mapping();

    // init the tracker
    if (pangolin::Pushed(_init)) {
      // sample points in the src and dst hands and create priors
      create_priors();
    }

    // save current hand pose as GT
    if (pangolin::Pushed(_save_result)) save_result("hand_pose.txt");

    if (pangolin::Pushed(_reset_joints)) {
      for (auto &slider: sliders) slider = 0.f;
    }

    // update src model pose according to sliders
    auto &pose  = tracker->getPose(src_hand_id);
    for (int j=0; j < sliders.size(); j++) {
      pose.getReducedArticulation()[j] = sliders[j].Get();
    }
    tracker->updatePose(src_hand_id);

    // rendering
    // models
    camDisp.ActivateScissorAndClear(camState);
    for (int m = 0; m < tracker->getNumModels(); ++m) {
      if (m == src_hand_id && !_show_src_hand) continue;
      if (m == dst_hand_id && !_show_dst_hand) continue;
      const auto &model = tracker->getModel(m);
      model.render();
      if (_show_axes)
        model.renderCoordinateFrames(0.05);
    }
    if (_show_priors) {
      // show contact points
      for (const auto &p: priors) {
        if (!p) continue;
        show_3d_3d_prior(p);
      }
    }

    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow(window_name);
  cout << "Exiting pangolin thread" << endl;
}


float3 joint_in_model(const dart::Model &model, int frame_id) {
  const auto &Tj  = model.getTransformFrameToModel(frame_id);
  float4 p = Tj * make_float4(0, 0, 0, 1);
  return make_float3(p.x, p.y, p.z);
}


void HandMapper::create_priors() {
    // find chain lengths
  vector<float> src_chain_lengths;
  const dart::Model &src_model = tracker->getModel(src_hand_id);
  int finger_id(-1);
  float chain_length(0.f);
  for (int f=1; f<src_model.getNumFrames(); f++) {
    if ((f-1)%4 == 0) {
      finger_id++;
      chain_length = 0.f;
    }
    if ((f-1)%4 == 3) {
      src_chain_lengths.push_back(chain_length);
      continue;
    }
    float3 axis = joint_in_model(src_model, f+1) - joint_in_model(src_model, f);
    chain_length += length(axis);
  }

  // sample points in the src model
  float dt;
  finger_id = -1;
  vector<vector<float3> > src_pts(src_chain_lengths.size());
  vector<float3> src_chain_bases(src_chain_lengths.size());
  for (int f=1; f<src_model.getNumFrames(); f++) {
    if ((f-1)%4 == 0) {
      finger_id++;
      dt = src_chain_lengths[finger_id] / n_priors;
    }
    if ((f-1)%4 == 3) continue;
    float3 base = joint_in_model(src_model, f);
    float3 axis = joint_in_model(src_model, f+1) - base;
    if (length(axis) < 5e-3) continue;
    int t(0);
    while (true) {
      float3 p = base + t*dt*normalize(axis);
      if (length(p-base) > length(axis)) break;
      // transform p to the dst coordinate system
      float4 pd = dart::SE3Invert(T_s_d) * make_float4(p, 1);
      p = make_float3(pd.x, pd.y, pd.z) / src_chain_lengths[finger_id];
      if (t == 0 && (f-1)%4==0) src_chain_bases[finger_id] = p;
      src_pts[finger_id].push_back(p);
      t++;
    }
  }

  // get targets for dst model
  // find chain lengths
  vector<float> dst_chain_lengths;
  const dart::Model &dst_model = tracker->getModel(dst_hand_id);
  finger_id = -1;
  chain_length = 0.f;
  for (int f=1; f<dst_model.getNumFrames(); f++) {
    if ((f-1)%4 == 0) {
      finger_id++;
      chain_length = 0.f;
    }
    if ((f-1)%4 == 3) {
      dst_chain_lengths.push_back(chain_length);
      continue;
    }
    float3 axis = joint_in_model(dst_model, f+1) - joint_in_model(dst_model, f);
    chain_length += length(axis);
  }

  // sample points in the dst model
  finger_id = -1;
  vector<vector<pair<float3, int> > > dst_pts(dst_chain_lengths.size());
  vector<vector<float3> > dst_targs(dst_chain_lengths.size());
  int src_counter(0), src_finger_id(-1);
  float3 base_offset = make_float3(0, 0, 0);
  for (int f=1; f<dst_model.getNumFrames(); f++) {
    float3 base = joint_in_model(dst_model, f);
    if ((f-1)%4 == 0) {
      finger_id++;
      src_finger_id++;
      if (finger_id == 3) src_finger_id++;
      src_counter = 0;
      dt = dst_chain_lengths[finger_id] / n_priors;
      base_offset = base - src_chain_bases[src_finger_id]*dst_chain_lengths[finger_id];
    }
    if ((f-1)%4 == 3) continue;

    float3 axis = joint_in_model(dst_model, f+1) - base;
    if (length(axis) < 5e-3) continue;
    int t(0);
    while (true) {
      float3 p = base + t*dt*normalize(axis);
      if (length(p-base) > length(axis)) break;
      if (src_counter == src_pts[src_finger_id].size()) break;

      float4 pf = dst_model.getTransformModelToFrame(f) * make_float4(p, 1);
      dst_pts[finger_id].push_back(std::make_pair(make_float3(pf.x, pf.y, pf.z), f));

      float3 src_p = src_pts[src_finger_id][src_counter]*dst_chain_lengths[finger_id];
      src_p += base_offset;
      float4 src_pc = dst_model.getTransformModelToCamera() * make_float4(src_p, 1);
      dst_targs[finger_id].push_back(make_float3(src_pc.x, src_pc.y, src_pc.z));
      t++;
      src_counter++;
    }
  }

  // create the priors
  tracker->clearPriors();
  priors.clear();
  for (finger_id=0; finger_id<dst_pts.size(); finger_id++) {
    for (int i=0; i<dst_pts[finger_id].size(); i++) {
      shared_ptr<dart::Point3D3DPrior> prior =
          std::make_shared<dart::Point3D3DPrior>(dst_hand_id,
              dst_pts[finger_id][i].second, dst_targs[finger_id][i],
              dst_pts[finger_id][i].first, _attract_weight);
      priors.push_back(prior);
      tracker->addPrior(prior);
    }
  }
}


float3 HandMapper::frame_point_in_camera(const float3 &frame_pt, int frame_id) {
  const auto &model = tracker->getModel(dst_hand_id);
  float4 src_pt_c = model.getTransformFrameToCamera(frame_id) *
      make_float4(frame_pt, 1);
  float3 src_pt = make_float3(src_pt_c.x, src_pt_c.y, src_pt_c.z);
  return src_pt;
}


void HandMapper::show_3d_3d_prior(const shared_ptr<dart::Point3D3DPrior> &prior,
    float size) {
  const float3 src_pt(frame_point_in_camera(prior->getSourceFramePoint(),
      prior->getSourceFrame())), dst_pt(prior->getTargetCameraPoint());
  glPointSize(size);

  // draw points
  glBegin(GL_POINTS);
  glColor3f(1, 0, 0);  // red for the joint locations
  glVertex3fv(&src_pt.x);
  glColor3f(0, 1, 0);  // green for the target points
  glVertex3fv(&dst_pt.x);
  glEnd();

  // draw lines connecting src and dst points
  glBegin(GL_LINES);
  glColor3f(1, 1, 1);  // red for the joint locations
  glVertex3fv(&src_pt.x);
  glVertex3fv(&dst_pt.x);
  glEnd();
}


void HandMapper::set_max_iterations(int m) {
  _max_iterations = m;
  tracker->getOptions().numIterations = m;
}

void HandMapper::set_attract_weight(float w) {
  _attract_weight = w;
  for (const auto &p: priors) {
    p->setWeight(_attract_weight / n_priors);
  }
}

void HandMapper::set_intra_model_intersection(float w) {
  _intra_model_intersection = w;
  tracker->getOptions().lambdaIntersection[
      dst_hand_id + tracker->getNumModels()*dst_hand_id] =
          _intra_model_intersection;
}

void HandMapper::set_inter_model_intersection(float w) {
  _inter_model_intersection = w;
  tracker->getOptions().lambdaIntersection[
      src_hand_id + tracker->getNumModels()*dst_hand_id] = _inter_model_intersection;
  tracker->getOptions().lambdaIntersection[
      dst_hand_id + tracker->getNumModels()*src_hand_id] = _inter_model_intersection;
}

void HandMapper::set_log10_regularization(float w) {
  _log10_regularization = w;
  std::fill(tracker->getOptions().regularization.begin(),
      tracker->getOptions().regularization.end(),
      powf(10.f, _log10_regularization));
  tracker->getOptions().contactRegularization =
      powf(10.f, _log10_regularization);
}

void HandMapper::set_lm_damping(float w) {
  _lm_damping = w;
  std::fill(tracker->getOptions().regularizationScaled.begin(),
      tracker->getOptions().regularizationScaled.end(), _lm_damping);
  tracker->getOptions().contactRegularizationScaled = _lm_damping;
}


int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "Usage ./" << argv[0] << " instruction src_model.xml dst_model.xml "
                                     "[graspit]"
        << endl;
    return -1;
  }
  bool graspit(false);
  if (argc == 5 && string(argv[4]) == "graspit") graspit = true;
  string instruction(argv[1]);
  string src_model_filename(argv[2]), dst_model_filename(argv[3]);

  stringstream filename;
  filename << "grasps/" << instruction << "_grasps.txt";
  ifstream f(filename.str());
  if (!f.is_open()) {
    cout << "Could not open " << filename.str() << " for reading" << endl;
    return -1;
  }

  HandMapper hm(src_model_filename, dst_model_filename, graspit);
  string line;
  while(getline(f, line)) {
    stringstream ss(line);
    string object_name, session_num;
    ss >> object_name >> session_num;
    stringstream ssf;
    if (graspit) ssf << "graspit_";
    ssf << session_num << "_" << instruction << "_" << object_name
        << "_gt_hand_pose.txt";

    hm.set_src_pose(ssf.str());
    hm.do_mapping();
    hm.save_result(ssf.str());
  }
  f.close();

  return 0;
}