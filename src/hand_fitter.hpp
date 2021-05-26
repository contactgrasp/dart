//
// Created by samarth on 7/25/18.
//

#ifndef DART_HAND_FITTER_HPP
#define DART_HAND_FITTER_HPP

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <boost/filesystem.hpp>

#include <string>
#include <thread>
#include <condition_variable>
#include <unordered_map>

// DART includes
#include "tracker_no_obs.h"
#include "util/gl_dart.h"
#include "optimization/priors.h"
#include "hand_fitter_config.h"

class HandFitter {
 public:
  HandFitter(const std::string &data_dir_=MODELS_DIR);
  ~HandFitter();

  void show_axes(bool show) { _show_axes = show; }
  void close() { pangolin::Quit(); }
  void set_targets(const std::unordered_map<std::string, float3>
    &joint_locations, const std::unordered_map<std::string, bool>
    &inner_surface);
  Eigen::Matrix4f init();
  dart::Pose fit(float &residue);

  void set_max_iterations(int m);
  void set_attract_weight(float w);
  void set_intra_model_intersection(float w);
  void set_lm_damping(float w);
  void set_log10_regularization(float w);
  void set_params(float aw, float iw, float lmd, float reg) {
    set_attract_weight(aw);
    set_intra_model_intersection(iw);
    set_lm_damping(lmd);
    set_log10_regularization(reg);
  }

 private:
  bool _init, _fit, _show_axes, _show_hand, _calc_error, _print_grasp;
  int _max_iterations;
  float _attract_weight, _intra_model_intersection, _lm_damping,
      _log10_regularization;
  std::shared_ptr<std::thread> pango_thread;
  boost::filesystem::path data_dir;

  std::shared_ptr<dart::TrackerNoObs> tracker;
  bool tracker_ready;
  std::mutex tracker_mutex;
  std::condition_variable tracker_cv;

  std::vector<int> graspit2dartdofs;

  // stores the tacker priors
  std::unordered_map<std::string, std::shared_ptr<dart::Point3D3DPrior> >
      contact_priors;
  // stores whether joint was observed on outer/inner (0/1) hand surface
  std::unordered_map<std::string, size_t> joint2inout;
  // stores 3D point locations for each joint
  // each joint has 2 3D points: outer/inner hand surface
  std::unordered_map<std::string, std::vector<float3> > joint2framept;
  // stores the frame ID of each joint
  std::unordered_map<std::string, size_t> joint2frameid;
  // stores joint target points (in camera coordinate frame)
  std::unordered_map<std::string, float3> joint_targets;
  // randomly sampled DOF initializations
  std::vector<std::vector<float> > dof_inits;

  Eigen::Matrix4f fit_rigid_joints();
  void pangolin_loop();
  void create_pangolin_vars_callbacks();
  void create_tracker();
  void show_3d_3d_prior(const std::string &joint_name, float size=10.f);
  float3 src_frame_point_in_camera(const std::string &joint_name);
  float calculate_error();
  Eigen::Matrix4f estimate_srt(Eigen::MatrixXf src_pts, Eigen::MatrixXf dst_pts);
};

#endif //DART_HAND_FITTER_HPP
