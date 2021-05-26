//
// Created by samarth on 7/25/18.
//

#ifndef DART_GRASP_ANALYZER_HPP
#define DART_GRASP_ANALYZER_HPP

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

#include <atomic>
#include <string>
#include <thread>
#include <memory>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <boost/filesystem.hpp>

// DART includes
#include "tracker_no_obs.h"
#include "util/gl_dart.h"
#include "optimization/priors.h"
#include "pose/pose_reduction.h"

class GraspAnalyser {
 public:
  GraspAnalyser(const std::string &object_name, const std::string &session_name,
      const std::string &hand_model,
      bool allegro_mapped=false, bool graspit=false,
      const std::string &data_dir=std::string("grasps"),
      float scale=1.426976f);  // usually 1.15

  ~GraspAnalyser() {
    if (!pango_thread) return;
    if (pango_thread->joinable()) pango_thread->join();
    if (pose_reduction) delete pose_reduction;
  }

  void show_axes(bool show) { _show_axes = show; }
  void sort_energy(bool sort);
  void sort_gt(bool sort);
  void save_error() { _calc_error = true; }
  void close() { quit_pango_loop = true; }
  // TODO implement other actions

  float analyze_grasps(int n, int order_by=-1, int energy_type=0,
      bool called_from_pango_thread=false);

  void set_order(int col_idx=0);
  void reset_order();

  void set_max_iterations(int m);
  void set_attract_weight(float w);
  void set_repulse_weight(float w);
  void set_thumb_attract_weight(float w);
  void set_attract_dist_cm(float d);
  void set_repulse_dist_cm(float d);
  void set_inter_model_intersection(float w);
  void set_intra_model_intersection(float w);
  void set_lm_damping(float w);
  void set_log10_regularization(float w);
  void set_params(float aw, float rw, float tw, float ad, float rd, float iw,
      float lmd, float reg) {
    set_attract_weight(aw);
    set_repulse_weight(rw);
    set_thumb_attract_weight(tw);
    set_attract_dist_cm(ad);
    set_repulse_dist_cm(rd);
    set_inter_model_intersection(iw);
    set_intra_model_intersection(iw);
    set_lm_damping(lmd);
    set_log10_regularization(reg);
  }

 private:
  std::string object_name, session_name;

  // pangolin variables
  bool _iterate, _init, _show_axes, _sort_energy, _sort_gt, _save_gt,
      _calc_error, _analyze_grasps,
      _show_priors, _show_hand, _show_object;
  bool quit_pango_loop;
  bool graspit;
  int _max_iterations, _grasp_idx, _analysis_energy;
  float _attract_weight, _repulse_weight, _thumb_attract_weight;
  float _attract_dist_cm, _repulse_dist_cm;
  float _inter_model_intersection, _intra_model_intersection;
  float _lm_damping, _log10_regularization;
  float scale;
  std::shared_ptr<std::thread> pango_thread;

  std::shared_ptr<dart::TrackerNoObs> tracker;
  int hand_id, object_id;
  std::mutex tracker_mutex;
  std::condition_variable tracker_cv;
  bool tracker_ready;
  bool analyze_from_pango_thread;

  std::vector<dart::Pose> grasps;
  std::vector<int> graspit2dartdofs;
  std::vector<int> order;

  std::vector<dart::SE3> gt_frame_poses;

  std::vector<std::shared_ptr<dart::ContactPrior> > contact_priors;
  std::shared_ptr<dart::ContactPrior> thumb_contact_prior;
  size_t n_attract_points, n_repulse_points, thumb_frame_id;

  boost::filesystem::path data_dir, hand_model_filename;

  std::queue<std::pair<int, int> > analysis_q;
  std::atomic_bool analysis_q_ready;
  std::deque<std::vector<float> > result_q;
  std::atomic_bool result_q_ready;
  bool save_result_q;

  float3 thumb_contact_point;
  std::string hand_name;

  // pose reduction is only needed for Barrett
  dart::LinearPoseReduction *pose_reduction;

  void pangolin_loop();
  void create_pangolin_vars_callbacks();
  void load_object();
  void create_tracker();
  void save_results();
};

#endif //DART_GRASP_ANALYZER_HPP
