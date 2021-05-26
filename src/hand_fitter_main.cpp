//
// Created by samarth on 6/25/18.
//
#include "hand_fitter.hpp"

using namespace std;

int main(int argc, char **argv) {
  HandFitter hf;
  unordered_map<string, float3> targets;
  targets["palm"] = make_float3(-0.259641, -0.210706, 0.851218);
  targets["thumb1"] = make_float3(-0.239734, -0.21468, 0.874517);
  targets["thumb2"] = make_float3(-0.209842, -0.22252, 0.897368);
  targets["thumb3"] = make_float3(-0.184112, -0.224282, 0.915278);
  targets["thumb4"] = make_float3(-0.159927, -0.214221, 0.926451);
  targets["index1"] = make_float3(-0.20367, -0.261045, 0.874487);
  targets["index2"] = make_float3(-0.172356, -0.276262, 0.878248);
  targets["index3"] = make_float3(-0.151159, -0.267969, 0.879777);
  targets["index4"] = make_float3(-0.137874, -0.256043, 0.879069);
  targets["mid1"] = make_float3(-0.205182, -0.262663, 0.854478);
  targets["mid2"] = make_float3(-0.165002, -0.26504, 0.855362);
  targets["mid3"] = make_float3(-0.143703, -0.247693, 0.857613);
  targets["mid4"] = make_float3(-0.131786, -0.23264, 0.861327);
  targets["ring1"] = make_float3(-0.207011, -0.255679, 0.838354);
  targets["ring2"] = make_float3(-0.170365, -0.257052, 0.830792);
  targets["ring3"] = make_float3(-0.151288, -0.239911, 0.836148);
  targets["ring4"] = make_float3(-0.139203, -0.226595, 0.840471);
  targets["pinky1"] = make_float3(-0.208756, -0.245651, 0.82259);
  targets["pinky2"] = make_float3(-0.180924, -0.241683, 0.817133);
  targets["pinky3"] = make_float3(-0.164308, -0.231857, 0.821239);
  targets["pinky4"] = make_float3(-0.152415, -0.225227, 0.823129);
  unordered_map<string, bool> inner_surface;
  for (const auto &p: targets) inner_surface[p.first] = false;
  hf.set_targets(targets, inner_surface);
  return 0;
}
