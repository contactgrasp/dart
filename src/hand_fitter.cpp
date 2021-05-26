//
// Created by samarth on 7/25/18.
//

#define CUDA_ERR_CHECK

#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <algorithm>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>

// DART
#include "util/dart_io.h"
#include "util/ostream_operators.h"
#include "hand_fitter.hpp"

// for Umeyama transform
#include <Eigen/Geometry>

using namespace std;
namespace fs = boost::filesystem;


float3 HandFitter::src_frame_point_in_camera(const string &joint_name) {
  const auto &model = tracker->getModel(0);
  const int frameNum = joint2frameid[joint_name];
  const size_t frame_pt_idx = joint2inout[joint_name];
  float4 src_pt_c = model.getTransformFrameToCamera(frameNum) *
      make_float4(joint2framept[joint_name][frame_pt_idx], 1);
  float3 src_pt = make_float3(src_pt_c.x, src_pt_c.y, src_pt_c.z);
  return src_pt;
}


void HandFitter::show_3d_3d_prior(const string &joint_name, float size) {
  const auto &prior = contact_priors[joint_name];
  const float3 src_pt(src_frame_point_in_camera(joint_name)),
      dst_pt(prior->getTargetCameraPoint());
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

string string_from_pose(const dart::Pose &pose) {
  stringstream f;
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
  return f.str();
}

HandFitter::HandFitter(const std::string &data_dir_) :
    _init(false), _show_axes(false), _show_hand(true), _calc_error(false),
    _print_grasp(false),
    _fit(false),
    _max_iterations(40),
    _attract_weight(20.f),
    _intra_model_intersection(20.f),
    _lm_damping(60.f), _log10_regularization(-1.f),
    tracker_ready(false),
    data_dir(data_dir_) {
  // mapping from graspit DOFs to dart DOFs
  graspit2dartdofs.resize(20);
  iota(graspit2dartdofs.begin(), graspit2dartdofs.end(), 0);

  // start the pangolin thread
  pango_thread = make_shared<thread>(&HandFitter::pangolin_loop, this);

  // wait for tracker to be ready
  if (!tracker_ready) {
    unique_lock<mutex> lk(tracker_mutex);
    tracker_cv.wait(lk, [this]{return this->tracker_ready;});
  }

  // map from frame name to ID
  unordered_map<string, int> geom_name2id;
  geom_name2id["palm"] = 0;
  geom_name2id["index1"] = 2;
  geom_name2id["index2"] = 3;
  geom_name2id["index3"] = 4;
  geom_name2id["mid1"] = 6;
  geom_name2id["mid2"] = 7;
  geom_name2id["mid3"] = 8;
  geom_name2id["ring1"] = 10;
  geom_name2id["ring2"] = 11;
  geom_name2id["ring3"] = 12;
  geom_name2id["pinky1"] = 14;
  geom_name2id["pinky2"] = 15;
  geom_name2id["pinky3"] = 16;
  geom_name2id["thumb1"] = 18;
  geom_name2id["thumb2"] = 19;
  geom_name2id["thumb3"] = 20;
  fs::path kp_filename = data_dir / "HumanHand" / "keypoints.txt";
  ifstream f(kp_filename.string());
  if (!f.is_open()) {
    cout << "Could not open " << kp_filename << endl;
    return;
  }
  string line;
  while (getline(f, line)) {
    stringstream ss(line);
    string geom_name, joint_name;
    float x, y, z, offset_x, offset_y, offset_z;
    ss >> joint_name >> geom_name >> x >> y >> z;

    // get offsets
    fs::path offset_filename = data_dir / "HumanHand" / "meshes" / geom_name /
        "offset.txt";
    ifstream of(offset_filename.string());
    if (!of.is_open()) {
      cout << "Could not open " << offset_filename << endl;
      return;
    }
    of >> offset_x >> offset_y >> offset_z;
    of.close();

    x = (x + offset_x) / 1000.f;
    y = (y + offset_y) / 1000.f;
    z = (z + offset_z) / 1000.f;

    joint2frameid[joint_name] = geom_name2id[geom_name];
    joint2framept[joint_name].push_back(make_float3(x, y, z));  // outer surface
    joint2framept[joint_name].push_back(make_float3(0, 0, 0));  // inner surface
  }
  f.close();

  // read dof initializations
  fs::path dof_init_filename = data_dir / ".." / "grasps" / "dof_samples.txt";
  f.open(dof_init_filename.string());
  if (!f.is_open()) {
    cout << "Could not open " << dof_init_filename << " for reading" << endl;
    return;
  }
  while (getline(f, line)) {
    stringstream ss(line);
    dof_inits.emplace_back();
    for (int i=0; i<tracker->getPose(0).getReducedArticulatedDimensions(); i++) {
      float dof_val;
      char comma;
      ss >> dof_val >> comma;
      dof_inits.back().push_back(dof_val);
    }
  }
  cout << "Read " << dof_inits.size() << " DOF initializations." << endl;
}

HandFitter::~HandFitter() {
  if (!pango_thread) return;
  if (pango_thread->joinable()) pango_thread->join();
}


void HandFitter::set_targets(const std::unordered_map<string, float3>
    &joint_locations, const std::unordered_map<string, bool> &inner_surface) {
  float3 origin = make_float3(0, 0, 0);
  if (joint_locations.count("palm")) origin = joint_locations.at("palm");
  joint_targets.clear();
  joint2inout.clear();
  for (const auto &p: joint_locations) {
    // cout << "Setting target for " << p.first << ": " << p.second << endl;
    joint_targets[p.first] = p.second;
    joint2inout[p.first] = inner_surface.at(p.first) ? 1 : 0;
  }
}


Eigen::Matrix4f HandFitter::estimate_srt(Eigen::MatrixXf src,
    Eigen::MatrixXf dst) {
  Eigen::Matrix4f best_T;
  size_t n_iters(25), n_sample_pts(3);
  float inlier_thresh(1e-3);
  if (src.cols() < n_sample_pts) {
    cout << "Need at least " << n_sample_pts << " points, have " << src.cols()
    << endl;
    return best_T;
  }
  if (src.rows() != 3 || dst.rows() != 3) {
    cout << "Input data should have 3 rows" << endl;
    return best_T;
  }

  // make homogeneous
  Eigen::MatrixXf h_src = Eigen::MatrixXf::Constant(src.rows()+1, src.cols(), 1);
  h_src.topRows(3) = src;
  Eigen::MatrixXf h_dst = Eigen::MatrixXf::Constant(dst.rows()+1, dst.cols(), 1);
  h_dst.topRows(3) = dst;

  vector<size_t> best_inliers;
  vector<size_t> order(src.cols());
  std::iota(order.begin(), order.end(), 0);
  for (int i=0; i<n_iters; i++) {
    // cout << "Iteration " << i << endl;
    // sample randomly
    random_shuffle(order.begin(), order.end());
    Eigen::Matrix<float, 3, Eigen::Dynamic> src_pts, dst_pts;
    src_pts.resize(3, n_sample_pts);
    dst_pts.resize(3, n_sample_pts);
    for (int j=0; j<n_sample_pts; j++) {
      src_pts.col(j) = src.col(order[j]);
      dst_pts.col(j) = dst.col(order[j]);
    }

    // estimate model
    Eigen::Matrix4f T = Eigen::umeyama(src_pts, dst_pts, true);

    // count inliers
    Eigen::MatrixXf src_dst = T * h_src;
    Eigen::MatrixXf dists = src_dst.topRows(3) - h_dst.topRows(3);
    dists = dists.array().square();
    dists = dists.colwise().sum().eval();  // avoid aliasing
    vector<size_t> inliers;
    for (int j=0; j<dists.size(); j++) {
      if (dists(j) < inlier_thresh) {
        inliers.push_back(j);
      }
    }

    // update best stats
    if (inliers.size() > best_inliers.size()) {
      best_inliers = inliers;
      // cout << "Better than before: " << inliers.size() << " inliers" << endl;
    }
  }

  // estimate final model with all inliers
  Eigen::Matrix<float, 3, Eigen::Dynamic> src_pts, dst_pts;
  src_pts.resize(3, best_inliers.size());
  dst_pts.resize(3, best_inliers.size());
  for (int j=0; j<best_inliers.size(); j++) {
    src_pts.col(j) = src.col(best_inliers[j]);
    dst_pts.col(j) = dst.col(best_inliers[j]);
  }

  // estimate model
  Eigen::Matrix4f T = Eigen::umeyama(src_pts, dst_pts, true);

  return T;
}


Eigen::Matrix4f HandFitter::fit_rigid_joints() {
  // create point clouds for ICP
  Eigen::Matrix<float, 3, Eigen::Dynamic> src_pts, dst_pts;
  vector<string> rigid_joints;
  for (const auto &p: joint_targets) {
    // use only points on the palm
    if (joint2frameid[p.first] == 0) rigid_joints.push_back(p.first);
  }

  src_pts.resize(3, rigid_joints.size());
  dst_pts.resize(3, rigid_joints.size());
  size_t idx(0);
  for (const string &joint_name: rigid_joints) {
    float3 tp = src_frame_point_in_camera(joint_name);
    dst_pts.col(idx) = Eigen::Vector3f(tp.x, tp.y, tp.z);
    // cout << "Target point " << tp << endl;
    float3 sp = joint_targets[joint_name];
    src_pts.col(idx) = Eigen::Vector3f(sp.x, sp.y, sp.z);
    // cout << "Source point " << sp << endl;
    idx++;
  }

  // perform Umeyama transform
  Eigen::Matrix4f T = estimate_srt(src_pts, dst_pts);
  return T;
}


Eigen::Matrix4f HandFitter::init() {
  // zero out all dofs
  auto &pose = tracker->getPose(0);
  pose.setTransformModelToCamera(dart::SE3());
  for (int i=0; i<pose.getReducedArticulatedDimensions(); i++) {
    pose.getReducedArticulation()[i] = 0.f;
  }
  tracker->updatePose(0);

  // fit rigid palm points
  Eigen::Matrix4f T = fit_rigid_joints();
  cout << "Scale = " << T.block<3, 3>(0, 0).determinant() << endl;

  // create / set the priors
  for (const auto &p: joint_targets) {
    const string &joint_name(p.first);
    const float3 &frame_pt = joint2framept[joint_name][joint2inout[joint_name]];

    float3 camera_pt = joint_targets[joint_name];
    // transform the target points using the rigid transform
    Eigen::Vector4f camera_pt_T = T * Eigen::Vector4f(camera_pt.x, camera_pt.y,
        camera_pt.z, 1);
    camera_pt.x = camera_pt_T(0);
    camera_pt.y = camera_pt_T(1);
    camera_pt.z = camera_pt_T(2);

    if (contact_priors.count(joint_name)) {  // this joint's prior already exists
      contact_priors[joint_name]->setTargetCameraPoint(camera_pt);
    } else {  // prior needs to be constructed
      auto prior =
          make_shared<dart::Point3D3DPrior>(0, joint2frameid[joint_name],
              camera_pt, frame_pt, _attract_weight);
      contact_priors[joint_name] = prior;
      tracker->addPrior(prior);
    }
  }

  return T;
}


dart::Pose HandFitter::fit(float &residue) {
  dart::Pose best_pose = tracker->getPose(0);
  float best_error(1000.f);
  for (const auto &dofs: dof_inits) {
    dart::Pose &pose = tracker->getPose(0);
    pose.setTransformModelToCamera(dart::SE3());
    for (int i=0; i<pose.getReducedArticulatedDimensions(); i++) {
      pose.getReducedArticulation()[i] = dofs[i];
    }
    tracker->updatePose(0);
    tracker->optimizePoses();
    float error = calculate_error();
    if (error < best_error) {
      best_error = error;
      best_pose = tracker->getPose(0);
    }
  }
  cout << "Fit with best error = " << best_error << endl;
  tracker->getPose(0) = best_pose;
  tracker->updatePose(0);
  return best_pose;
}


float HandFitter::calculate_error() {
  float error(0.f);
  int count(0);
  for (const auto &p: contact_priors) {
    const float3 &src_pt = src_frame_point_in_camera(p.first);
    const float3 &dst_pt = p.second->getTargetCameraPoint();
    error += length(src_pt - dst_pt);
    count++;
  }
  return error / count;
}


void HandFitter::create_tracker() {
  tracker = make_shared<dart::TrackerNoObs>();
  fs::path hand_model_filename = data_dir / "HumanHand" / "human_hand.xml";
  tracker->addModel(hand_model_filename.string());
  fs::path ipot_filename = data_dir / "HumanHand" /
      "intersection_potential_matrix.txt";

  // zero out the hand-object intersection matrix
  int n = tracker->getNumModels();
  memset(tracker->getOptions().lambdaIntersection.data(), 0, n*n*sizeof(float));

  // set hand self-intersection matrix
  int *hand_self_intersection = dart::loadSelfIntersectionMatrix(
      ipot_filename.string(), tracker->getModel(0).getNumSdfs());
  tracker->setIntersectionPotentialMatrix(0, hand_self_intersection);
  delete [] hand_self_intersection;
}

void HandFitter::create_pangolin_vars_callbacks() {
  // setup control buttons and sliders
  pangolin::Var<bool>::Attach("ui.init", _init, false);
  pangolin::Var<bool>::Attach("ui.fit", _fit, false);
  pangolin::Var<bool>::Attach("ui.show axes", _show_axes, true);
  pangolin::Var<bool>::Attach("ui.show hand", _show_hand, true);
  pangolin::Var<bool>::Attach("ui.calculate error", _calc_error, false);
  pangolin::Var<int> ::Attach("ui.max # iterations", _max_iterations, 0, 100);
  pangolin::Var<float>::Attach("ui.attraction weight", _attract_weight, 0, 400);
  pangolin::Var<float>::Attach("ui.intra model intersection",
      _intra_model_intersection, 0, 100);
  pangolin::Var<float>::Attach("ui.LM damping", _lm_damping, 0, 100);
  pangolin::Var<float>::Attach("ui.log10(regularization)",
      _log10_regularization, -10, 0);

  // callback functions for the sliders
  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandFitter *ga = (HandFitter *)data;
        pangolin::Var<int> v(_var);
        ga->set_max_iterations(v.Get());
      },
      (void *)this, "ui.max # iterations");
  set_max_iterations(_max_iterations);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandFitter *ga = (HandFitter *)data;
        pangolin::Var<float> v(_var);
        ga->set_attract_weight(v.Get());
      },
      (void *)this, "ui.attraction weight");
  set_attract_weight(_attract_weight);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandFitter *ga = (HandFitter *)data;
        pangolin::Var<float> v(_var);
        ga->set_intra_model_intersection(v.Get());
      },
      (void *)this, "ui.intra model intersection");
  set_intra_model_intersection(_intra_model_intersection);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandFitter *ga = (HandFitter *)data;
        pangolin::Var<float> v(_var);
        ga->set_lm_damping(v.Get());
      },
      (void *)this, "ui.LM damping");
  set_lm_damping(_lm_damping);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        HandFitter *ga = (HandFitter *)data;
        pangolin::Var<float> v(_var);
        ga->set_log10_regularization(v.Get());
      },
      (void *)this, "ui.log10(regularization)");
  set_log10_regularization(_log10_regularization);
}

void HandFitter::pangolin_loop() {
  // OpenGL context needs to be created inside the thread that runs the
  // Pangolin window
  const float im_width(1920), im_height(1200), cam_focal_length(420),
      ui_width(0.3f);
  cudaGLSetGLDevice(0);
  cudaDeviceReset();
  string window_name("Hand Fitter");
  pangolin::CreateWindowAndBind(window_name, im_width, im_height);
  glewInit();

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

  while(!pangolin::ShouldQuit()) {
    // calculate the error of current pose
    if (pangolin::Pushed(_calc_error))
      cout << "Error = " << calculate_error() << endl;

    // init the tracker
    if (pangolin::Pushed(_init)) init();

    if (pangolin::Pushed(_fit)) {
      float residue;
      fit(residue);
    }

    // save current hand pose as GT
    if (pangolin::Pushed(_print_grasp)) {
      auto const &pose = tracker->getPose(0);
      cout << "Palm pose = " << endl << pose.getTransformCameraToModel() << endl
          << " Joints = " << endl;
      for (int i=0; i<pose.getArticulatedDimensions(); i++) {
        cout << pose.getArticulation()[i] << " ";
      }
      cout << endl;
    }

    // rendering
    // models
    camDisp.ActivateScissorAndClear(camState);
    for (int m = 0; m < tracker->getNumModels(); ++m) {
      if (_show_hand) {
        const auto &model = tracker->getModel(m);
        model.render();
        if (_show_axes)
          model.renderCoordinateFrames(0.05);
      }
    }
    // contact points
    for (const auto &p: contact_priors) {
      if (!p.second) continue;
      show_3d_3d_prior(p.first);
    }

    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow(window_name);
  cout << "Exiting pangolin thread" << endl;
}

void HandFitter::set_max_iterations(int m) {
  _max_iterations = m;
  tracker->getOptions().numIterations = m;
}

void HandFitter::set_attract_weight(float w) {
  _attract_weight = w;
  for (const auto &p: contact_priors) p.second->setWeight(_attract_weight);
}

void HandFitter::set_intra_model_intersection(float w) {
  _intra_model_intersection = w;
  tracker->getOptions().lambdaIntersection[0] = _intra_model_intersection;
}

void HandFitter::set_log10_regularization(float w) {
  _log10_regularization = w;
  std::fill(tracker->getOptions().regularization.begin(),
      tracker->getOptions().regularization.end(),
      powf(10.f, _log10_regularization));
  tracker->getOptions().contactRegularization =
      powf(10.f, _log10_regularization);
}

void HandFitter::set_lm_damping(float w) {
  _lm_damping = w;
  std::fill(tracker->getOptions().regularizationScaled.begin(),
      tracker->getOptions().regularizationScaled.end(), _lm_damping);
  tracker->getOptions().contactRegularizationScaled = _lm_damping;
}
