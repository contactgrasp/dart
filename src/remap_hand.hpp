//
// Created by samarth on 2/28/19.
//

#ifndef DART_REMAP_HAND_HPP
#define DART_REMAP_HAND_HPP

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <atomic>
#include <string>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <boost/filesystem.hpp>

// DART includes
#include "tracker_no_obs.h"
#include "util/gl_dart.h"
#include "optimization/priors.h"

class HandMapper {
 public:
  HandMapper(const std::string &src_model_filename,
      const std::string &dst_model_filename,
      bool graspit=false,
      const std::string &data_dir=std::string("grasps"));

  ~HandMapper() {
    if (!pango_thread) return;
    if (pango_thread->joinable()) pango_thread->join();
  }

  void show_axes(bool show) { _show_axes = show; }
  void close() { pangolin::Quit(); }

  // set the pose of the src hand from a file
  void set_src_pose(const std::string &pose_filename);
  // map the src pose to dst pose
  float do_mapping();
  // save result
  void save_result(const std::string &src_filename);

  void set_max_iterations(int m);
  void set_attract_weight(float w);
  void set_attract_dist_cm(float d);
  void set_inter_model_intersection(float w);
  void set_intra_model_intersection(float w);
  void set_lm_damping(float w);
  void set_log10_regularization(float w);
  void set_params(float aw, float ad, float iw, float lmd, float reg) {
    set_attract_weight(aw);
    set_attract_dist_cm(ad);
    set_inter_model_intersection(iw);
    set_intra_model_intersection(iw);
    set_lm_damping(lmd);
    set_log10_regularization(reg);
  }

 private:
  std::string src_model_filename, dst_model_filename;

  // pangolin variables
  bool _iterate, _init, _show_axes, _calc_error, _save_result, _reset_joints,
      _show_priors, _show_src_hand, _show_dst_hand, _show_object;
  bool graspit;
  bool quit_pangolin_loop;
  int _max_iterations;
  float _attract_weight;
  float _inter_model_intersection, _intra_model_intersection;
  float _lm_damping, _log10_regularization;
  std::shared_ptr<std::thread> pango_thread;

  // holds the dof values of the src hand
  std::vector<pangolin::Var<float> > sliders;

  std::shared_ptr<dart::TrackerNoObs> tracker;
  int src_hand_id, dst_hand_id;
  std::mutex tracker_mutex;
  std::condition_variable tracker_cv;
  bool tracker_ready;
  bool analyze_from_pango_thread;
  std::vector<int> graspit2dartdofs;

  std::vector<std::shared_ptr<dart::Point3D3DPrior> > priors;
  size_t n_priors;

  boost::filesystem::path data_dir, src_hand_model_filename,
      dst_hand_model_filename;

  // pose of dst w.r.t. src
  dart::SE3 T_s_d;

  void pangolin_loop();
  void create_pangolin_vars_callbacks();
  void create_tracker();
  void create_priors();
  void show_3d_3d_prior(const std::shared_ptr<dart::Point3D3DPrior> &p,
      float size=10.f);
  float3 frame_point_in_camera(const float3 &frame_pt, int frame_id);
};

#endif //DART_REMAP_HAND_HPP