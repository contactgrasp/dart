//
// Created by samarth on 6/25/18.
//

// Copyright (c) 2016, Lula Robotics Inc.  All rights reserved.

#define CUDA_ERR_CHECK

#include <iostream>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Pangolin
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>

// CUDA
#include <cuda_runtime.h>

// DART
#include "tracker_no_obs.h"
#include "util/dart_io.h"
#include "util/gl_dart.h"
#include "util/ostream_operators.h"
#include "util/string_format.h"
#include "pose/pose_reduction.h"

using namespace std;

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage ./" << argv[0] << " model.xml" << endl;
    return -1;
  }
  string model_filename(argv[1]);

  // init cuda and opengl context
  const float focal_len(420), im_width(1920), im_height(1200);
  cudaGLSetGLDevice(0);
  cudaDeviceReset();
  pangolin::CreateWindowAndBind("model", im_width, im_height);
  glewInit();

  // setup viewport
  const float ui_width(0.3f);
  pangolin::OpenGlRenderState camState(
      pangolin::ProjectionMatrixRDF_TopLeft(im_width, im_height, focal_len,
          focal_len, im_width/2.f, im_height/2.f, 0.01, 1000),
      pangolin::ModelViewLookAtRDF(0, -0.25, 0, 0, 0, 0, 0, 0, 1));
  pangolin::View &camDisp =
      pangolin::Display("cam")
          .SetBounds(0, 1, pangolin::Attach(ui_width), 1, -im_width/im_height)
          .SetHandler(new pangolin::Handler3D(camState));
  // lighting
  glShadeModel(GL_SMOOTH);
  float4 lightPosition =
      make_float4(normalize(make_float3(-0.4405, -0.5357, -0.619)), 0);
  GLfloat light_ambient[] = {0.3, 0.3, 0.3, 1.0};
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
  glLightfv(GL_LIGHT0, GL_POSITION, (float *)&lightPosition);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHT0);
  glEnable(GL_NORMALIZE);
  glEnable(GL_LIGHTING);
  glColor4ub(0xff, 0xff, 0xff, 0xff);
  glEnable(GL_COLOR_MATERIAL);

  // setup control buttons and sliders
  pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach(ui_width));

  // create tracker and add objects to it
  dart::TrackerNoObs tracker;
  dart::LinearPoseReduction *reduction = nullptr;
  if (model_filename.find("Barrett") != string::npos) {  // barrett uses a reduced pose
    reduction = new dart::LinearPoseReduction(8, 4);
  }
  tracker.addModel(model_filename, 0.005, 0.10, reduction);
  if (model_filename.find("Barrett") != string::npos) {
    int F(reduction->getFullDimensions()), R(reduction->getReducedDimensions());
    const auto & model = tracker.getModel(0);
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
    reduction->init(A.data(), b.data(), jointMins.data(), jointMaxs.data(),
        jointNames.data());
  }
  dart::MirroredVector<const uchar3 *> allSdfColors(tracker.getNumModels());
  for (int m = 0; m < tracker.getNumModels(); ++m) {
    allSdfColors.hostPtr()[m] = tracker.getModel(m).getDeviceSdfColors();
  }
  allSdfColors.syncHostToDevice();

  // create sliders
  auto &model = tracker.getModel(0);
  auto &pose  = tracker.getPose(0);
  vector<pangolin::Var<float> > sliders;
  for (int i=0; i < pose.getReducedArticulatedDimensions(); i++) {
    stringstream name;
    name << "ui.dof" << i;
    pangolin::Var<float> slider(name.str(), 0.f,
        pose.getReducedMin(i), pose.getReducedMax(i));
    sliders.push_back(slider);
  }
  pangolin::Var<bool> show_axes_button("ui.show axes", false, true);
  pangolin::Var<bool> reset_joints_button("ui.reset DOFs", false, false);

  // main loop
  while (!pangolin::ShouldQuit()) {

    // rendering
    // models
    camDisp.ActivateScissorAndClear(camState);

    if (pangolin::Pushed(reset_joints_button)) {
      for (auto &slider: sliders) slider = 0.f;
    }

    for (int m = 0; m < tracker.getNumModels(); ++m) {
      auto &model = tracker.getModel(m);
      auto &pose  = tracker.getPose(m);
      for (int j=0; j < sliders.size(); j++) {
        pose.getReducedArticulation()[j] = sliders[j].Get();
      }
      tracker.updatePose(m);
      model.render();
      if (show_axes_button)
        model.renderCoordinateFrames(0.03);
    }

    // Finish frame
    pangolin::FinishFrame();
  }

  return 0;
}