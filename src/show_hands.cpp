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

using namespace std;

int main(int argc, char **argv) {
  string human_model_filename("models/HumanHand/human_hand.xml");
  string allegro_model_filename("models/allegro/allegro.xml");

  // init cuda and opengl context
  const float focal_len(420), im_width(1920), im_height(1200);
  cudaGLSetGLDevice(0);
  cudaDeviceReset();
  pangolin::CreateWindowAndBind("model", im_width, im_height);
  glewInit();

  // setup viewport
  const float human_ui_width(0.15f);
  const float allegro_ui_width(0.15f);
  pangolin::OpenGlRenderState camState(
      pangolin::ProjectionMatrixRDF_TopLeft(im_width, im_height, focal_len,
          focal_len, im_width/2.f, im_height/2.f, 0.01, 1000),
      pangolin::ModelViewLookAtRDF(0, -0.25, 0, 0, 0, 0, 0, 0, 1));
  pangolin::View &camDisp =
      pangolin::Display("cam")
          .SetBounds(0, 1, pangolin::Attach(human_ui_width+allegro_ui_width), 1,
              -im_width/im_height)
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
  pangolin::CreatePanel("hui").SetBounds(0, 1, 0, pangolin::Attach(human_ui_width));
  pangolin::CreatePanel("aui").SetBounds(0, 1, pangolin::Attach(human_ui_width),
      pangolin::Attach(human_ui_width+allegro_ui_width));

  // create tracker and add objects to it
  dart::TrackerNoObs tracker;
  int human_id(0), allegro_id(1);
  tracker.addModel(human_model_filename);
  tracker.addModel(allegro_model_filename);
  dart::MirroredVector<const uchar3 *> allSdfColors(tracker.getNumModels());
  for (int m = 0; m < tracker.getNumModels(); ++m) {
    allSdfColors.hostPtr()[m] = tracker.getModel(m).getDeviceSdfColors();
  }
  allSdfColors.syncHostToDevice();

  // create sliders
  auto &model = tracker.getModel(human_id);
  vector<pangolin::Var<float> > human_sliders;
  for (int i=0; i < model.getNumJoints(); i++) {
    string name = string("hui.") + model.getJointName(i);
    float val = fmax(0.f, model.getJointMin(i));
    pangolin::Var<float> slider(name, val, model.getJointMin(i), model.getJointMax(i));
    human_sliders.push_back(slider);
  }
  pangolin::Var<bool> human_show_axes_button("hui.show axes", false, true);
  pangolin::Var<bool> human_reset_joints_button("hui.reset joints", false, false);

  auto &amodel = tracker.getModel(allegro_id);
  vector<pangolin::Var<float> > allegro_sliders;
  for (int i=0; i < amodel.getNumJoints(); i++) {
    string name = string("aui.") + amodel.getJointName(i);
    float val = fmax(0.f, amodel.getJointMin(i));
    pangolin::Var<float> slider(name, val, amodel.getJointMin(i), amodel.getJointMax(i));
    allegro_sliders.push_back(slider);
  }
  pangolin::Var<bool> allegro_show_axes_button("aui.show axes", false, true);
  pangolin::Var<bool> allegro_reset_joints_button("aui.reset joints", false, false);

  // poses
  dart::SE3 human_pose = dart::SE3FromRotationX(0);
  dart::SE3 T_h_a = dart::SE3FromTranslation(-0.14, 0, 0) *
      dart::SE3FromRotationY(-M_PI/2) * dart::SE3FromRotationZ(-M_PI/2);
  dart::SE3 allegro_pose = human_pose * T_h_a;
  tracker.getPose(human_id).setTransformModelToCamera(human_pose);
  tracker.updatePose(human_id);
  tracker.getPose(allegro_id).setTransformModelToCamera(allegro_pose);
  tracker.updatePose(allegro_id);

  // main loop
  while (!pangolin::ShouldQuit()) {

    // rendering
    // models
    camDisp.ActivateScissorAndClear(camState);

    if (pangolin::Pushed(human_reset_joints_button)) {
      for (auto &slider: human_sliders) slider = 0.f;
    }
    if (pangolin::Pushed(allegro_reset_joints_button)) {
      for (auto &slider: allegro_sliders) slider = 0.f;
    }

    for (int m = 0; m < tracker.getNumModels(); ++m) {
      auto &model = tracker.getModel(m);
      auto &pose  = tracker.getPose(m);
      vector<pangolin::Var<float> > &sliders = m==human_id ? human_sliders : allegro_sliders;
      for (int j=0; j < sliders.size(); j++) {
        pose.getReducedArticulation()[j] = sliders[j].Get();
      }
      tracker.updatePose(m);
      model.render();
      if (m==human_id && human_show_axes_button) model.renderCoordinateFrames(0.15);
      if (m==allegro_id && allegro_show_axes_button) model.renderCoordinateFrames(0.15);
    }

    // Finish frame
    pangolin::FinishFrame();
  }

  return 0;
}