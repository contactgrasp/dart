#!/usr/bin/env bash

GRASPS_DIR=${HOME}/contactgrasp_data/grasps
MODELS_DIR=${HONE}/deepgrasp_data/models

HUMAN_MESHES_DIR=${HOME}/research/thermal_grasp/data/human_hand_meshes

set -x
ln -s $GRASPS_DIR grasps
ln -s $MODELS_DIR models/object_models
ln -s $HUMAN_MESHES_DIR models/HumanHand/meshes
set +x