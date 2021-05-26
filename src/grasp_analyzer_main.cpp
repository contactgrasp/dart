//
// Created by samarth on 6/25/18.
//
#include "grasp_analyzer.hpp"

using namespace std;

int main(int argc, char **argv) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " object_name session_name "
                                    "hand_model_filename [no_analysis] "
                                    "[allegro_mapped] [graspit]" << endl;
    return -1;
  }
  bool do_analysis(true), allegro_mapped(false), graspit(false);
  if (argc >= 5 && string(argv[4]) == "no_analysis") do_analysis = false;
  if (argc >= 6 && string(argv[5]) == "allegro_mapped") allegro_mapped = true;
  if (argc >= 7 && string(argv[6]) == "graspit") graspit = true;

  GraspAnalyser ga(argv[1], argv[2], argv[3], allegro_mapped, graspit);
  if (do_analysis) ga.analyze_grasps(-1, -1, 0);
  return 0;
}
