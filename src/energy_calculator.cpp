//
// Created by samarth on 6/25/18.
//
#include "grasp_analyzer.hpp"

using namespace std;

int main(int argc, char **argv) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " object_name session_name "
                                    "hand_model_filename [allegro_mapped]"
                                    " [graspit]"
        << endl;
    return -1;
  }

  string object_name(argv[1]), session_name(argv[2]), model_filename(argv[3]);
  bool allegro_mapped(false), graspit(false);
  if (argc >= 5 && string(argv[4]) == "allegro_mapped") allegro_mapped = true;
  if (argc >= 6 && string(argv[5]) == "graspit") graspit = true;

  GraspAnalyser ga(object_name, session_name, model_filename, allegro_mapped,
      graspit);
  ga.set_inter_model_intersection(0);
  ga.set_intra_model_intersection(0);
  // ga.set_attract_weight(1);
  // ga.set_repulse_weight(1);
  ga.set_thumb_attract_weight(0);
  this_thread::sleep_for(chrono::milliseconds(100));
  ga.save_error();
  this_thread::sleep_for(chrono::milliseconds(100));
  ga.close();
  return 0;
}
