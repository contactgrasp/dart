//
// Created by samarth on 7/25/18.
//

#define CUDA_ERR_CHECK

#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <random>
#include <algorithm>
#include <cfloat>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA
#include <cuda_runtime.h>

// DART
#include "util/dart_io.h"
#include "grasp_analyzer.hpp"

using namespace std;
namespace fs = boost::filesystem;

void show_contact_prior(const shared_ptr<dart::ContactPrior> &prior,
    const dart::TrackerNoObs &tracker, vector<float> color={1,0,0},
    float size=10.f) {
  glPointSize(size);
  glBegin(GL_POINTS);
    auto &model = tracker.getModel(prior->getSourceModel());
    const int sdfNum = prior->getSourceSdfNum();
    const int frameNum = model.getSdfFrameNumber(sdfNum);
    const float3 contactPoint = prior->getContactPoint();
    float4 contact_c = model.getTransformFrameToCamera(frameNum) *
        make_float4(contactPoint, 1);
    glColor3f(color[0], color[1], color[2]);
    glVertex3fv(&contact_c.x);
  glEnd();
}

struct IncrementVarFunctor
{
  IncrementVarFunctor(const std::string& name, int maxval) : varName(name), maxval(maxval) {}
  void operator()() { pangolin::Var<int> val(varName); val=min(maxval, val+1);}
  std::string varName;
  int maxval;
};
struct DecrementVarFunctor
{
  DecrementVarFunctor(const std::string& name, int minval) : varName(name), minval(minval) {}
  void operator()() { pangolin::Var<int> val(varName); val=max(minval, val-1);}
  std::string varName;
  int minval;
};


void allegro_pose_from_human_string(const string &line, dart::Pose &pose,
    vector<int> dof_mapping) {
  stringstream ss(line);

  // parse the line
  vector<float> vals;
  char comma;
  float val;
  while (ss >> val) {
    vals.push_back(val);
    ss >> comma;
  }
  if (vals.size() < 12 + pose.getReducedArticulatedDimensions()) {
    cout << "Need " << 12+pose.getReducedArticulatedDimensions() << " dofs in"
         << " file, got only " << vals.size() << endl;
    return;
  }

  size_t hdof_idx(0);
  // read human hand pose
  vector<float4> palm_pose(3);
  for (int i = 0; i < palm_pose.size(); i++) {
    palm_pose[i].x = vals[hdof_idx]; hdof_idx++;
    palm_pose[i].y = vals[hdof_idx]; hdof_idx++;
    palm_pose[i].z = vals[hdof_idx]; hdof_idx++;
    palm_pose[i].w = vals[hdof_idx]; hdof_idx++;
  }
  dart::SE3 T_c_h = dart::SE3(palm_pose[0], palm_pose[1], palm_pose[2]);
  // pose of allegro w.r.t. hand
  dart::SE3 T_h_a = dart::SE3FromTranslation(-0.095, 0, 0) *
      dart::SE3FromRotationY(-M_PI/2) * dart::SE3FromRotationZ(-M_PI/2);
  pose.setTransformModelToCamera(T_c_h * T_h_a);
  // another way to get the same pose
  // dart::SE3 T_a_h = dart::SE3FromRotationY(M_PI/2) *
  // dart::SE3FromRotationX(-M_PI/2) * dart::SE3FromTranslation(0.1, 0, 0);
  // pose.setTransformModelToCamera(T_c_h * dart::SE3Invert(T_a_h));
  // read dofs
  size_t adof_idx;
  // first 4 fingers
  for (adof_idx=0; adof_idx < pose.getReducedArticulatedDimensions()-4;
      adof_idx++, hdof_idx++) {
    pose.getReducedArticulation()[dof_mapping[adof_idx]] = vals[hdof_idx];
  }
  // thumb needs special mapping
  hdof_idx += 4;
  pose.getReducedArticulation()[dof_mapping[adof_idx]] = -vals[hdof_idx] + 1.185;
  hdof_idx++; adof_idx++;
  pose.getReducedArticulation()[dof_mapping[adof_idx]] = 0.5422f * vals[hdof_idx+1];
  hdof_idx++; adof_idx++;
  pose.getReducedArticulation()[dof_mapping[adof_idx]] = vals[hdof_idx] +
      vals[hdof_idx-1] + 0.814f;
  hdof_idx++; adof_idx++;
  pose.getReducedArticulation()[dof_mapping[adof_idx]] = vals[hdof_idx];
}


void pose_from_string(const string &line, dart::Pose &pose,
    vector<int> dof_mapping, bool read_gt=false) {
  stringstream ss(line);

  // parse the line
  vector<float> vals;
  char comma;
  float val;
  if (read_gt) ss >> val >> comma;
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
    pose.getReducedArticulation()[dof_mapping[i]] = vals[dof_idx];
    dof_idx++;
  }
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

float compare_poses(const vector<dart::SE3> &p1, const vector<dart::SE3> &p2) {
  float dist = 0.f;
  for (int f=0; f < p1.size(); f++) {
    float3 t1(dart::translationFromSE3(p1[f])), t2(dart::translationFromSE3(p2[f]));
    dist += powf(t1.x-t2.x, 2.f) + powf(t1.y-t2.y, 2.f) + powf(t1.z-t2.z, 2.f);
  }
  // dist = sqrtf(dist);
  return dist;
}


GraspAnalyser::GraspAnalyser(const std::string &object_name,
    const std::string &session_name, const std::string &hand_model,
    bool allegro_mapped, bool graspit, const std::string &data_dir,
    float scale) :
    object_name(object_name), session_name(session_name), data_dir(data_dir),
    hand_model_filename(hand_model), graspit(graspit),
    _iterate(false), _init(false),
    _show_axes(false), _show_priors(false), _show_hand(true), _show_object(true),
    _calc_error(false),
    _sort_energy(false), _sort_gt(false),
    _save_gt(false), save_result_q(false),
    _analyze_grasps(false), _analysis_energy(0),
    _max_iterations(40),
    _grasp_idx(0),
    _attract_weight(150.f), _repulse_weight(20.f), _thumb_attract_weight(25.f),
    _attract_dist_cm(2.f), _repulse_dist_cm(2.f),
    _inter_model_intersection(100.f), _intra_model_intersection(5.f),
    _lm_damping(25.f), _log10_regularization(-7.5f),
    hand_id(0), object_id(1),
    tracker_ready(false), analysis_q_ready(false), result_q_ready(false),
    analyze_from_pango_thread(false), quit_pango_loop(false),
    pose_reduction(nullptr),
    scale(scale)
{
  // check if hand is allgro
  hand_name = string("human");
  if (allegro_mapped) hand_name = "allegro_mapped";
  else if (hand_model_filename.string().find("allegro") != string::npos) {
    hand_name = string("allegro");
  } else if (hand_model_filename.string().find("Barrett") != string::npos) {
    hand_name = string("barrett");
  }

  if (hand_name == "allegro") {
    _attract_dist_cm = 5.f;
  }
  if (hand_name == "barrett") {
    _log10_regularization = -1.f;
  }

  // read location of thumb contact point
  fs::path thumb_filename = hand_model_filename.parent_path() / "thumb.txt";
  ifstream f(thumb_filename.string());
  if (!f.is_open()) {
    cout << "Could not read thumb filename " << thumb_filename << endl;
    return;
  }
  f >> thumb_frame_id >> thumb_contact_point.x >> thumb_contact_point.y
      >> thumb_contact_point.z;
  f.close();

  // start the pangolin thread
  pango_thread = make_shared<thread>(&GraspAnalyser::pangolin_loop, this);

  // wait for tracker to be ready
  if (!tracker_ready) {
    unique_lock<mutex> lk(tracker_mutex);
    tracker_cv.wait(lk, [this]{return this->tracker_ready;});
  }
}


void GraspAnalyser::load_object() {
  if (!tracker) {
    cerr << "Call create_tracker() first before calling load_object()" << endl;
    return;
  }

  string obj_base_name = object_name.substr(0, object_name.find_first_of('-'));
  tracker->addModel(string("models/object_models/") + obj_base_name +
      string(".xml"));
  dart::MirroredVector<const uchar3 *> allSdfColors(tracker->getNumModels());
  for (int m = 0; m < tracker->getNumModels(); ++m) {
    allSdfColors.hostPtr()[m] = tracker->getModel(m).getDeviceSdfColors();
  }
  allSdfColors.syncHostToDevice();

  // load grasps
  fs::path grasp_filename;
  if (hand_name == "allegro" || hand_name == "allegro_mapped")
    grasp_filename = data_dir / (obj_base_name+string("_grasps_allegro.csv"));
  else if (hand_name == "barrett")
    grasp_filename = data_dir / (obj_base_name+string("_grasps_barrett.csv"));
  else grasp_filename = data_dir / (obj_base_name+string("_grasps.csv"));
  ifstream f(grasp_filename.string());
  if (!f.is_open()) {
    cerr << "Could not open " << grasp_filename << " for reading" << endl;
    return;
  }
  string line;
  getline(f, line);  // ignore first line (comment)
  while (getline(f, line)) {
    dart::Pose pose(tracker->getPose(hand_id));
    pose_from_string(line, pose, graspit2dartdofs);
    grasps.push_back(pose);
  }
  f.close();
  cout << "Loaded " << grasps.size() << " grasps from " << grasp_filename << endl;
  if (grasps.empty()) return;
  reset_order();
  pangolin::Var<int>::Attach("ui.grasp idx", _grasp_idx, -1, grasps.size()-1);
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL+'n',
      DecrementVarFunctor("ui.grasp idx", -1));
  pangolin::RegisterKeyPressCallback(pangolin::PANGO_CTRL+'m',
      IncrementVarFunctor("ui.grasp idx", grasps.size()-1));

  // read the ground truth pose
  stringstream ss;
  if (graspit) ss << "graspit_";
  if (hand_name == "allegro_mapped") ss << "allegro_mapped_";
  else if (hand_name == "allegro") ss << "allegro_";
  else if (hand_name == "barrett") ss << "barrett_";
  ss << session_name << "_" << object_name << "_gt_hand_pose.txt";
  fs::path gt_filename = data_dir / ss.str();
  f.open(gt_filename.string());
  if (!f.is_open()) {
    cerr << "No GT hand pose file at " << gt_filename << endl;
  } else {
    dart::Pose gt_pose(grasps[0]);
    getline(f, line);
    pose_from_string(line, gt_pose, graspit2dartdofs, true);
    cout << "Read GT hand pose from " << gt_filename << endl;
    tracker->getPose(hand_id) = gt_pose;
    tracker->updatePose(hand_id);
    const auto &model = tracker->getModel(hand_id);
    // works because object pose is identity
    for (int i=0; i < model.getNumFrames(); i++)
      gt_frame_poses.push_back(model.getTransformFrameToCamera(i));
    f.close();
  }

  // read contact information (points and normals)
  scale /= 1.15;  // to remove the scale factor that already exists
  ss.str(""); ss.clear();
  ss << session_name << "_" << object_name << "_contact_info.txt";
  fs::path contact_filename = data_dir / ss.str();
  f.open(contact_filename.string());
  if (!f.is_open()) {
    cerr << "Could not open " << contact_filename << " for reading." << endl;
    return;
  }
  vector<float3> contact_points, contact_normals, no_contact_points,
      no_contact_normals;
  while (getline(f, line)) {
    stringstream ss(line);
    int is_contact;
    float px, py, pz, nx, ny, nz;
    ss >> is_contact >> px >> py >> pz >> nx >> ny >> nz;
    px *= scale;
    py *= scale;
    pz *= scale;
    if (is_contact) {
      contact_points.push_back(make_float3(px, py, pz));
      contact_normals.push_back(make_float3(nx, ny, nz));
    } else {
      no_contact_points.push_back(make_float3(px, py, pz));
      no_contact_normals.push_back(make_float3(nx, ny, nz));
    }
  }
  f.close();
  n_attract_points = min(contact_points.size(), 500);
  // shuffle contact information
  auto rng = default_random_engine {};
  vector<size_t> idx(contact_points.size());
  iota(idx.begin(), idx.end(), 0);
  shuffle(idx.begin(), idx.end(), rng);
  idx.resize(n_attract_points);
  // create ContactPriors for contact points
  for (auto i: idx) {
    // normals will not be used since these are contact points
    shared_ptr<dart::ContactPrior> contact_prior =
        make_shared<dart::ContactPrior>(object_id, hand_id, 0, -1,
            _attract_weight/n_attract_points, contact_points[i],
            false, true, true, false, _attract_dist_cm/100.f,
            contact_normals[i]);
    tracker->addPrior(contact_prior);
    contact_priors.push_back(contact_prior);
  }
  // create ContactPriors for non-contact points
  n_repulse_points = min(no_contact_points.size(), 2000);
  idx.resize(no_contact_points.size());
  iota(idx.begin(), idx.end(), 0);
  shuffle(idx.begin(), idx.end(), rng);
  idx.resize(n_repulse_points);
  for (auto i: idx) {
    shared_ptr<dart::ContactPrior> contact_prior =
        make_shared<dart::ContactPrior>(object_id, hand_id, 0, -1,
            _repulse_weight/n_attract_points, no_contact_points[i],
            false, true, true, true, _repulse_dist_cm/100.f,
            no_contact_normals[i]);
    tracker->addPrior(contact_prior);
    contact_priors.push_back(contact_prior);
  }
  // thumb
  // don't need to provide surface normal, since this is a contact point
  thumb_contact_prior = make_shared<dart::ContactPrior>(hand_id, object_id,
      thumb_frame_id, 0, _thumb_attract_weight, thumb_contact_point,
      false, true, true, false, _attract_dist_cm/100.f);
  tracker->addPrior(thumb_contact_prior);

  // zero out the hand-object intersection matrix
  int n = tracker->getNumModels();
  memset(tracker->getOptions().lambdaIntersection.data(), 0, n*n*sizeof(float));
}


void GraspAnalyser::create_tracker() {
  if (hand_name == "barrett") {
    pose_reduction = new dart::LinearPoseReduction(8, 4);
  }
  tracker = make_shared<dart::TrackerNoObs>();
  tracker->addModel(hand_model_filename.string(), 0.005, 0.10, pose_reduction);
  if (hand_name == "barrett") {
    int F(pose_reduction->getFullDimensions()),
        R(pose_reduction->getReducedDimensions());
    const auto & model = tracker->getModel(hand_id);
    std::vector<float> jointMins, jointMaxs;
    std::vector<std::string> jointNames;
    for (int j=0; j<model.getNumJoints(); j++) {
      jointMins.push_back(model.getJointMin(j));
      jointMaxs.push_back(model.getJointMax(j));
      jointNames.push_back(model.getJointName(j));
    }
    vector<float> A(F*R, 0), b(F, 0);
    A[0*R + 0] = 1;
    A[1*R + 1] = 1;
    A[2*R + 1] = 1;
    A[3*R + 0] = 1;
    A[4*R + 2] = 1;
    A[5*R + 2] = 1;
    A[6*R + 3] = 1;
    A[7*R + 3] = 1;
    pose_reduction->init(A.data(), b.data(), jointMins.data(), jointMaxs.data(),
        jointNames.data());
  }
  fs::path self_intersection_filename = hand_model_filename.parent_path() /
    "intersection_potential_matrix.txt";
  int *hand_self_intersection = dart::loadSelfIntersectionMatrix(
      self_intersection_filename.string(), tracker->getModel(hand_id).getNumSdfs());
  tracker->setIntersectionPotentialMatrix(hand_id, hand_self_intersection);
  delete [] hand_self_intersection;

  // mapping from graspit DOFs to dart DOFs
  graspit2dartdofs.resize(
      tracker->getPose(hand_id).getReducedArticulatedDimensions());
  // mapping for the modified graspit human hand
  // int mapping[20] = {1,  0,  2,  3, 5,  4,  6,  7, 9,  8,  10, 11,
  //     13, 12, 14, 15, 16, 17, 18, 19};
  // identity mapping
  iota(graspit2dartdofs.begin(), graspit2dartdofs.end(), 0);
}


void GraspAnalyser::create_pangolin_vars_callbacks() {
  // setup control buttons and sliders
  pangolin::Var<bool>::Attach("ui.init", _init, false);
  pangolin::Var<bool>::Attach("ui.iterate", _iterate, false);
  pangolin::Var<bool>::Attach("ui.show axes", _show_axes, true);
  pangolin::Var<bool>::Attach("ui.show contact priors", _show_priors, true);
  pangolin::Var<bool>::Attach("ui.show hand", _show_hand, true);
  pangolin::Var<bool>::Attach("ui.show object", _show_object, true);
  pangolin::Var<bool>::Attach("ui.sort grasps (energy)", _sort_energy, true);
  pangolin::Var<bool>::Attach("ui.sort grasps (GT)", _sort_gt, true);
  pangolin::Var<bool>::Attach("ui.save GT", _save_gt, false);
  pangolin::Var<bool>::Attach("ui.calculate error", _calc_error, false);
  pangolin::Var<int>::Attach("ui.max # iterations", _max_iterations, 0, 80);
  pangolin::Var<float>::Attach("ui.attraction weight", _attract_weight, 0, 400);
  pangolin::Var<float>::Attach("ui.repulsion weight", _repulse_weight, 0, 400);
  pangolin::Var<float>::Attach("ui.thumb attraction weight", _thumb_attract_weight,
      0, 100);
  pangolin::Var<float>::Attach("ui.attraction distance (cm)",
      _attract_dist_cm, 0, 5);
  pangolin::Var<float>::Attach("ui.repulsion distance (cm)",
      _repulse_dist_cm, 0, 5);
  pangolin::Var<float>::Attach("ui.inter model intersection",
      _inter_model_intersection, 0, 100);
  pangolin::Var<float>::Attach("ui.intra model intersection",
      _intra_model_intersection, 0, 100);
  pangolin::Var<float>::Attach("ui.LM damping", _lm_damping, 0, 100);
  pangolin::Var<float>::Attach("ui.log10(regularization)",
      _log10_regularization, -10, 0);
  pangolin::Var<int>::Attach("ui.analysis energy", _analysis_energy, -1, 1);
  pangolin::Var<bool>::Attach("ui.Analyze Grasps", _analyze_grasps, false);

  // callback functions for the sliders
  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<int> v(_var);
        ga->set_max_iterations(v.Get());
      },
      (void *)this, "ui.max # iterations");
  set_max_iterations(_max_iterations);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_attract_weight(v.Get());
      },
      (void *)this, "ui.attraction weight");
  set_attract_weight(_attract_weight);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_repulse_weight(v.Get());
      },
      (void *)this, "ui.repulsion weight");
  set_repulse_weight(_repulse_weight);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_thumb_attract_weight(v.Get());
      },
      (void *)this, "ui.thumb attraction weight");
  set_thumb_attract_weight(_thumb_attract_weight);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_attract_dist_cm(v.Get());
      },
      (void *)this, "ui.attraction distance (cm)");
  set_attract_dist_cm(_attract_dist_cm);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_repulse_dist_cm(v.Get());
      },
      (void *)this, "ui.repulsion distance (cm)");
  set_repulse_dist_cm(_repulse_dist_cm);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_inter_model_intersection(v.Get());
      },
      (void *)this, "ui.inter model intersection");
  set_inter_model_intersection(_inter_model_intersection);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_intra_model_intersection(v.Get());
      },
      (void *)this, "ui.intra model intersection");
  set_intra_model_intersection(_intra_model_intersection);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_lm_damping(v.Get());
      },
      (void *)this, "ui.LM damping");
  set_lm_damping(_lm_damping);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<float> v(_var);
        ga->set_log10_regularization(v.Get());
      },
      (void *)this, "ui.log10(regularization)");
  set_log10_regularization(_log10_regularization);

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<bool> v(_var);
        if (v) {
          ga->set_order(0);
          cout << "Sorted by energy" << endl;
        } else ga->reset_order();
      },
      (void *)this, "ui.sort grasps (energy)");

  pangolin::RegisterGuiVarChangedCallback(
      [](void *data, const std::string &name, pangolin::VarValueGeneric &_var) {
        GraspAnalyser *ga = (GraspAnalyser *)data;
        pangolin::Var<bool> v(_var);
        if (v) {
          ga->set_order(1);
          cout << "Sorted by GT similarity" << endl;
        } else ga->reset_order();
      },
      (void *)this, "ui.sort grasps (GT)");
}

void GraspAnalyser::pangolin_loop() {
  // OpenGL context needs to be created inside the thread that runs the
  // Pangolin window
  const float im_width(1920), im_height(1200), cam_focal_length(420),
      ui_width(0.3f);
  cudaGLSetGLDevice(0);
  cudaDeviceReset();
  string window_name("Grasp Analyzer");
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
  load_object();  // same for loading object meshes
  create_pangolin_vars_callbacks();

  // signal that the tracker is ready
  unique_lock<mutex> lk(tracker_mutex);
  tracker_ready = true;
  lk.unlock();
  tracker_cv.notify_one();

  while(!(pangolin::ShouldQuit() || quit_pango_loop)) {
    if (!analysis_q_ready) {
      // calculate the error of current pose
      if (pangolin::Pushed(_calc_error)) {
        float error = tracker->getError();
        cout << object_name << " Error = " << error << endl;
        stringstream ss;
        if (graspit) ss << "graspit_";
        if (hand_name == "allegro_mapped") ss << "allegro_mapped_";
        else if (hand_name == "allegro") ss << "allegro_";
        else if (hand_name == "barrett") ss << "barrett_";
        ss << session_name << "_" << object_name << "_grasp_error.txt";
        fs::path filename = data_dir / ss.str();
        ofstream f(filename.string());
        if (!f.is_open()) {
          cout << "Could not open " << filename.string() << " for writing"
              << endl;
          break;
        }
        f << error << endl;
        f.close();
        cout << "Written error to " << filename.string() << endl;
      }

      // optimize the poses
      if (pangolin::Pushed(_iterate)) {
        float error = tracker->optimizePoses();
        cout << "Error = " << error << endl;
      }

      // init the tracker
      if (pangolin::Pushed(_init)) {
        // hand
        int idx = order[max(0, _grasp_idx)];
        tracker->getPose(hand_id) = grasps[idx];
        tracker->updatePose(hand_id);
        cout << "Initialized by pose " << idx << endl;
        // for (int d=0; d < tracker->getPose(hand_id).getReducedArticulatedDimensions(); d++) {
        //   cout << tracker->getPose(hand_id).getReducedArticulation()[d] << " ";
        // }
        // cout << endl;

        // object
        dart::Pose &object_pose = tracker->getPose(object_id);
        object_pose.setTransformModelToCamera(dart::SE3());
        object_pose.getReducedArticulation()[0] = 0.f;
        tracker->updatePose(object_id);

        // thumb contact point
        float *p = thumb_contact_prior->getPriorParams();
        p[0] = thumb_contact_point.x;
        p[1] = thumb_contact_point.y;
        p[2] = thumb_contact_point.z;
      }

      // analyze all poses
      if (pangolin::Pushed(_analyze_grasps)) {
        analyze_grasps(-1, -1, _analysis_energy, true);
      }

      // save current hand pose as GT
      if (pangolin::Pushed(_save_gt)) {
        auto const &hand_pose = tracker->getPose(hand_id);
        auto const &object_pose = tracker->getPose(object_id);
        dart::Pose gt_pose(hand_pose);
        gt_pose.setTransformModelToCamera(
            object_pose.getTransformCameraToModel() *
                hand_pose.getTransformModelToCamera());
        stringstream ss;
        if (graspit) ss << "graspit_";
        if (hand_name == "allegro_mapped") ss << "allegro_mapped_";
        else if (hand_name == "allegro") ss << "allegro_";
        else if (hand_name == "barrett") ss << "barrett_";
        ss << session_name << "_" << object_name << "_gt_hand_pose.txt";
        fs::path gt_filename = data_dir / ss.str();
        ofstream f(gt_filename.string());
        if (f.is_open()) {
          f << _grasp_idx << "," << string_from_pose(gt_pose);
          f.close();
          cout << "GT hand pose written to " << gt_filename << endl;
        } else {
          cerr << "Could not open " << gt_filename << " for writing" << endl;
        }
      }
    } else {
      // process one grasp from the analysis queue
      const auto &p = analysis_q.front();
      analysis_q.pop();
      int idx = p.first;
      // type specifies the type of energy to be calculated
      // 0: DART, 1: GT similarity, -1: all
      int type = p.second;
      tracker->getPose(hand_id) = grasps[idx];
      tracker->updatePose(hand_id);
      tracker->getPose(object_id).setTransformModelToCamera(dart::SE3());
      tracker->getPose(object_id).getReducedArticulation()[0] = 0.f;
      tracker->updatePose(object_id);
      float *thumb_p = thumb_contact_prior->getPriorParams();
      thumb_p[0] = thumb_contact_point.x;
      thumb_p[1] = thumb_contact_point.y;
      thumb_p[2] = thumb_contact_point.z;
      float error = tracker->optimizePoses();
      tracker->updatePose(hand_id);
      tracker->updatePose(object_id);

      vector<float> result;
      if (type <= 0) result.push_back(error);

      if (type == 2 || type < 0) {
        const auto hand_pose = tracker->getPose(hand_id);
        const auto oTc = tracker->getPose(object_id).getTransformCameraToModel();
        // optimized_poses[idx] = hand_pose;
        // optimized_poses[idx].setTransformModelToCamera(
        //     oTc * hand_pose.getTransformModelToCamera());
        vector<dart::SE3> frame_poses;
        const auto &model = tracker->getModel(hand_id);
        for (int i = 0; i < gt_frame_poses.size(); i++)
          frame_poses.push_back(oTc * model.getTransformFrameToCamera(i));
        error = compare_poses(frame_poses, gt_frame_poses);
        result.push_back(error);
      }
      cout << "Grasp " << idx << " Results = ";
      for (const auto &r: result) cout << r << " ";
      cout << endl;
      result_q.push_back(result);
      if (analysis_q.empty()) {
        if (save_result_q) save_results();
        analysis_q_ready = false;
        result_q_ready = true;
        if (!analyze_from_pango_thread) break;
      }
    }

    // rendering
    // models
    camDisp.ActivateScissorAndClear(camState);
    for (int m = 0; m < tracker->getNumModels(); ++m) {
      if (m == hand_id && !_show_hand) continue;
      if (m == object_id && !_show_object) continue;
      const auto &model = tracker->getModel(m);
      model.render();
      if (_show_axes)
        model.renderCoordinateFrames(0.05);
    }
    if (_show_priors) {
      // show contact points
      for (const auto &p: contact_priors) {
        if (!p) continue;
        show_contact_prior(p, *tracker,
            {float(p->isInverted()), float(!p->isInverted()), 0.f});
      }
      if (thumb_contact_prior && _show_hand) {
        show_contact_prior(thumb_contact_prior, *tracker, {1, 1, 0});
      }
    }

    pangolin::FinishFrame();
  }
  pangolin::DestroyWindow(window_name);
  cout << "Exiting pangolin thread" << endl;
}

void GraspAnalyser::sort_energy(bool sort) {
  if (sort) {
    if (!_sort_energy) {
      _sort_gt = false;
      _sort_energy = true;
      set_order(0);
      cout << "Sorted by energy" << endl;
    } else cout << "Already sorted by energy" << endl;
  } else reset_order();
}

void GraspAnalyser::sort_gt(bool sort) {
  if (sort) {
    if (!_sort_gt) {
      _sort_energy = false;
      _sort_gt = true;
      set_order(1);
      cout << "Sorted by GT similarity" << endl;
    } else cout << "Already sorted by GT similarity" << endl;
  } else reset_order();
}

void GraspAnalyser::set_order(int col_idx) {
  stringstream ss;
  if (hand_name == "allegro" || hand_name == "allegro_mapped") ss << "allegro_";
  else if (hand_name == "barrett") ss << "barrett_";
  ss << session_name << "_" << object_name << "_grasp_errors.csv";
  fs::path errors_filename = data_dir / ss.str();
  ifstream f(errors_filename.string());
  if (!f.is_open()) {
    cout << "Could not open " << errors_filename << " for reading" << endl;
  } else {
    vector<float> errors(grasps.size(), FLT_MAX);
    string line;
    char comma;
    float value;
    size_t idx(0);
    while(getline(f, line)) {
      if (idx >= errors.size()) {
        cerr << "Errors file " << errors_filename << " has " << idx
             << " entries, should have " << grasps.size() << endl;
        break;
      }
      stringstream ss(line);
      for (int i=0; i < col_idx; i++) ss >> value >> comma;
      ss >> value;
      errors[idx] = value;
      idx++;
    }
    f.close();
    if (idx < grasps.size()) {
      cerr << "Errors file " << errors_filename << " has " << idx
           << " entries, should have " << grasps.size() << endl;
    }
    std::sort(order.begin(), order.end(),
        [&errors](size_t i1, size_t i2) { return errors[i1] < errors[i2]; });
  }
}

float GraspAnalyser::analyze_grasps(int n, int order_by, int energy_type,
    bool called_from_pango_thread) {
  analyze_from_pango_thread = called_from_pango_thread;
  // flush the analysis and result queues
  while (!analysis_q.empty()) {}
  while (!result_q.empty()) {result_q.pop_front();}

  // set flags
  result_q_ready = false;
  analysis_q_ready = false;
  save_result_q = true;

  // set the order
  if (order_by >= 0) set_order(order_by);
  else reset_order();

  // push the requests
  if (n < 0) n = grasps.size();
  if (energy_type == 0 || energy_type < 0)
    cout << "Analyzing DART energy" << endl;
  if (energy_type == 1 || energy_type < 0)
    cout << "Analyzing GT similarity" << endl;
  cout << "Pushing tasks...";
  for (int i=0; i < n; i++) analysis_q.push(make_pair(order[i], energy_type));
  analysis_q_ready = true;
  cout << " done" << endl;

  // if this function was called by clicking the UI button in Pangolin,
  // we need to exit this function after pushing the tasks
  // so that the pangolin_loop() function can process those tasks
  if (analyze_from_pango_thread) return -1.f;

  // if this function was called from the main thread, we need to hang out here
  // till the tasks are processed, so that the main thread does not exit
  // before all the tasks are processed in the pangolin thread
  // tasks are always processed in the pangolin thread, by the pango_loop() fn
  while (!result_q_ready) {
    this_thread::sleep_for(chrono::milliseconds(30));
  }

  // return the mean energy value for some legacy meta-optimization
  energy_type = max(0, energy_type);
  float mean(0.f);
  size_t N = result_q.size();
  while (!result_q.empty()) {
    vector<float> &r = result_q.front();
    mean += r[energy_type];
    result_q.pop_front();
  }
  mean /= N;
  return mean;
}

void GraspAnalyser::save_results() {
  stringstream ss;
  if (graspit) ss << "graspit_";
  if (hand_name == "allegro_mapped") ss << "allegro_mapped_";
  else if (hand_name == "allegro") ss << "allegro_";
  else if (hand_name == "barrett") ss << "barrett_";
  ss << session_name << "_" << object_name << "_grasp_errors.csv";
  fs::path errors_filename = data_dir / ss.str();
  ofstream f(errors_filename.string());
  if (f.is_open()) {
    for (const auto &result: result_q) {
      std::stringstream ss;
      for (const auto &rr: result) {
        ss << rr << ",";
      }
      std::string s(ss.str());
      f << s.substr(0, s.length()-1) << endl;
    }
    f.close();
    cout << "Saved results to " << errors_filename.string() << endl;
  } else {
    cerr << "Could not open " << errors_filename.string() << " for writing"
         << endl;
  }
}

void GraspAnalyser::reset_order() {
  order.resize(grasps.size());
  iota(order.begin(), order.end(), 0);
  cout << "No sorting" << endl;
}

void GraspAnalyser::set_max_iterations(int m) {
  _max_iterations = m;
  tracker->getOptions().numIterations = m;
}

void GraspAnalyser::set_attract_weight(float w) {
  _attract_weight = w;
  for (const auto &p: contact_priors) {
    if (!p->isInverted()) {
      // p->setWeight(_attract_weight / n_attract_points);
      p->setWeight(_attract_weight);
    }
  }
}

void GraspAnalyser::set_repulse_weight(float w) {
  _repulse_weight = w;
  for (const auto &p: contact_priors) {
    if (p->isInverted()) {
      // p->setWeight(_repulse_weight / n_attract_points);
      p->setWeight(_repulse_weight);
    }
  }
}

void GraspAnalyser::set_thumb_attract_weight(float w) {
  _thumb_attract_weight = w;
  thumb_contact_prior->setWeight(_thumb_attract_weight);
}

void GraspAnalyser::set_attract_dist_cm(float d) {
  _attract_dist_cm = d;
  for (const auto &p: contact_priors)
    if (!p->isInverted())
      p->setContactThreshold(_attract_dist_cm / 100.f);
  thumb_contact_prior->setContactThreshold(_attract_dist_cm / 100.f);
}

void GraspAnalyser::set_repulse_dist_cm(float d) {
  _repulse_dist_cm = d;
  for (const auto &p: contact_priors)
    if (p->isInverted())
      p->setContactThreshold(_repulse_dist_cm / 100.f);
}

void GraspAnalyser::set_intra_model_intersection(float w) {
  _intra_model_intersection = w;
  tracker->getOptions().lambdaIntersection[
      hand_id + tracker->getNumModels()*hand_id] = _intra_model_intersection;
}

void GraspAnalyser::set_inter_model_intersection(float w) {
  _inter_model_intersection = w;
  tracker->getOptions().lambdaIntersection[
      object_id + tracker->getNumModels()*hand_id] = _inter_model_intersection;
  tracker->getOptions().lambdaIntersection[
      hand_id + tracker->getNumModels()*object_id] = _inter_model_intersection;
}

void GraspAnalyser::set_log10_regularization(float w) {
  _log10_regularization = w;
  std::fill(tracker->getOptions().regularization.begin(),
      tracker->getOptions().regularization.end(),
      powf(10.f, _log10_regularization));
  tracker->getOptions().contactRegularization =
      powf(10.f, _log10_regularization);
}

void GraspAnalyser::set_lm_damping(float w) {
  _lm_damping = w;
  std::fill(tracker->getOptions().regularizationScaled.begin(),
      tracker->getOptions().regularizationScaled.end(), _lm_damping);
  tracker->getOptions().contactRegularizationScaled = _lm_damping;
}

