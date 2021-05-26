#include "tracker_no_obs.h"

#include "mesh/mesh_proc.h"
#include "mesh/mesh_sample.h"
#include "mesh/primitive_meshing.h"
#include "util/cuda_utils.h"
#include "util/dart_io.h"
#include "util/string_format.h"
#if ASSIMP_BUILD
#include "mesh/assimp_mesh_reader.h"
#endif // ASSIMP_BUILD
#if CUDA_BUILD
#include <cuda_gl_interop.h>
#endif // CUDA_BUILD

#include <GL/glx.h>

namespace dart {

TrackerNoObs::TrackerNoObs() : _optimizer(0),
    _T_mcs(0), _T_fms(0), _sdfFrames(0), _sdfs(0), _nSdfs(0), _distanceThresholds(0),
    _normalThresholds(0), _planeOffsets(0), _planeNormals(0), _dampingMatrices(0) {

    glewInit();

#if ASSIMP_BUILD
    Model::initializeRenderer(new AssimpMeshReader());
#else
    Model::initializeRenderer();
#endif // ASSIMP_BUILD

    cudaGLSetGLDevice(0);
    cudaDeviceReset();

    _optimizer = new OptimizerNoObs();
}

TrackerNoObs::~TrackerNoObs() {

    for (int m=0; m<_mirroredModels.size(); ++m) {
        delete _mirroredModels[m];
    }

    for (PoseReduction * reduction : _ownedPoseReductions) {
        delete reduction;
    }

    for (Eigen::MatrixXf * matrix : _dampingMatrices) {
        delete matrix;
    }

    delete _optimizer;

    Model::shutdownRenderer();

}

bool TrackerNoObs::addModel(const std::string & filename,
                       const float modelSdfResolution,
                       const float modelSdfPadding,
                       PoseReduction * poseReduction,
                       const float collisionCloudDensity,
                       const bool cacheSdfs) {

    HostOnlyModel model;
    if (!readModelXML(filename.c_str(),model)) {
        return false;
    }

    model.computeStructure();

    std::cout << "loading model from " << filename << std::endl;

    const int lastSlash = filename.find_last_of('/');
    const int lastDot = filename.find_last_of('.');
    const int diff = lastDot - lastSlash - 1;
    const int substrStart = lastSlash < filename.size() ? lastSlash + 1: 0;
    const std::string modelName = filename.substr(substrStart, diff > 0 ? diff : filename.size() - substrStart);
    std::cout << "model name: " << modelName << std::endl;

//    // TODO
//    if (model.getNumGeoms() < 2) {
//        model.voxelize2(modelSdfResolution,modelSdfPadding,cacheSdfs ? dart::stringFormat("model%02d",_mirroredModels.size()) : "");
//    } else {
        model.voxelize(modelSdfResolution,modelSdfPadding,cacheSdfs ? dart::stringFormat("/tmp/%s",modelName.c_str()) : "");
//    }

    _mirroredModels.push_back(new MirroredModel(model, make_uint3(0), 0));

    _sizeParams.push_back(model.getSizeParams());

    int nDimensions = model.getPoseDimensionality();
    if (poseReduction == 0) {
        std::vector<float> jointMins, jointMaxs;
        std::vector<std::string> jointNames;
        for (int j=0; j<model.getNumJoints(); ++j) {
            jointMins.push_back(model.getJointMin(j));
            jointMaxs.push_back(model.getJointMax(j));
            jointNames.push_back(model.getJointName(j));
        }
        poseReduction = new NullReduction(nDimensions - 6,
                                          jointMins.data(),
                                          jointMaxs.data(),
                                          jointNames.data());
        _ownedPoseReductions.push_back(poseReduction);
    }
    _estimatedPoses.push_back(Pose(poseReduction));

    _filenames.push_back(filename);

    // build collision cloud
    MirroredVector<float4> * collisionCloud = 0;
    _collisionCloudSdfLengths.push_back(std::vector<int>(model.getNumSdfs()));
    _collisionCloudSdfStarts.push_back(std::vector<int>(model.getNumSdfs()));
    int nPointsCumulative = 0;
    for (int f=0; f<model.getNumFrames(); ++f) {
        int sdfNum = model.getFrameSdfNumber(f);
        if (sdfNum >= 0) {
            _collisionCloudSdfStarts[getNumModels()-1][sdfNum] = nPointsCumulative;
            _collisionCloudSdfLengths[getNumModels()-1][sdfNum] = 0;
        }
        for (int g=0; g<model.getFrameNumGeoms(f); ++g) {
            int gNum = model.getFrameGeoms(f)[g];
            const SE3 mT = model.getGeometryTransform(gNum);
            const float3 scale = model.getGeometryScale(gNum);
            std::vector<float3> sampledPoints;
            Mesh * samplerMesh;
            switch (model.getGeometryType(gNum)) {
            case MeshType:
            {
                int mNum = model.getMeshNumber(gNum);
                const Mesh & mesh = model.getMesh(mNum);
                samplerMesh = new Mesh(mesh);
            }
                break;
            case PrimitiveSphereType:
                samplerMesh = generateUnitIcosphereMesh(2);
                break;
            case PrimitiveCylinderType:
                samplerMesh = generateCylinderMesh(30);
                break;
            case PrimitiveCubeType:
                samplerMesh = generateCubeMesh();
                break;
            default:
            {
                std::cerr << "collision clouds for type " << model.getGeometryType(gNum) << " not supported yet" << std::endl;
                continue;
            }
                break;
            }

            scaleMesh(*samplerMesh,scale);
            transformMesh(*samplerMesh,mT);
            sampleMesh(sampledPoints,*samplerMesh,collisionCloudDensity);
            delete samplerMesh;
//            std::cout << "sampled " << sampledPoints.size() << " points" << std::endl;

            int start;
            if (collisionCloud == 0) {
                start = 0;
                collisionCloud = new MirroredVector<float4>(sampledPoints.size());
            } else {
                start = collisionCloud->length();
                collisionCloud->resize(start + sampledPoints.size());
            }
            for (int v=0; v<sampledPoints.size(); ++v) {
                float4 vert = make_float4(sampledPoints[v],sdfNum);
                collisionCloud->hostPtr()[start + v] = vert;
            }

            _collisionCloudSdfLengths[getNumModels()-1][sdfNum] += sampledPoints.size();
            nPointsCumulative += sampledPoints.size();

        }
    }
    collisionCloud->syncHostToDevice();
    _collisionClouds.push_back(collisionCloud);
    MirroredVector<int> * intersectionPotentialMatrix = new MirroredVector<int>(model.getNumSdfs()*model.getNumSdfs());
    memset(intersectionPotentialMatrix->hostPtr(),0,intersectionPotentialMatrix->length()*sizeof(int));
    intersectionPotentialMatrix->syncHostToDevice();
    _intersectionPotentialMatrices.push_back(intersectionPotentialMatrix);

    if (getNumModels() == 1) {
        _T_mcs = new MirroredVector<SE3>(1);
        _T_fms = new MirroredVector<SE3*>(1);
        _sdfFrames = new MirroredVector<int*>(1);
        _sdfs = new MirroredVector<const Grid3D<float>*>(1);
        _nSdfs = new MirroredVector<int>(1);
        _distanceThresholds = new MirroredVector<float>(1);
        _normalThresholds = new MirroredVector<float>(1);
        _planeOffsets = new MirroredVector<float>(1);
        _planeNormals = new MirroredVector<float3>(1);
    } else {
        _T_mcs->resize(getNumModels());
        _T_fms->resize(getNumModels());
        _sdfFrames->resize(getNumModels());
        _sdfs->resize(getNumModels());
        _nSdfs->resize(getNumModels());
        _distanceThresholds->resize(getNumModels());
        _normalThresholds->resize(getNumModels());
        _planeOffsets->resize(getNumModels());
        _planeNormals->resize(getNumModels());
    }

    MirroredModel & mm = *_mirroredModels.back();
    const int m = getNumModels()-1;
    _T_fms->hostPtr()[m] = mm.getDeviceTransformsModelToFrame(); _T_fms->syncHostToDevice();
    _sdfFrames->hostPtr()[m] = mm.getDeviceSdfFrames(); _sdfFrames->syncHostToDevice();
    _sdfs->hostPtr()[m] = mm.getDeviceSdfs(); _sdfs->syncHostToDevice();
    _nSdfs->hostPtr()[m] = mm.getNumSdfs(); _nSdfs->syncHostToDevice();

    const int reducedDims = _estimatedPoses.back().getReducedDimensions();
    _dampingMatrices.push_back(new Eigen::MatrixXf(reducedDims,reducedDims));
    *_dampingMatrices.back() = Eigen::MatrixXf::Zero(reducedDims,reducedDims);

    _estimatedPoses.back().zero();
    _estimatedPoses.back().projectReducedToFull();
    mm.setPose(_estimatedPoses.back());

    if (getNumModels() > 1) {
        _opts.distThreshold.resize(getNumModels());         _opts.distThreshold.back() = _opts.distThreshold.front();
        _opts.regularization.resize(getNumModels());        _opts.regularization.back() = _opts.regularization.front();
        _opts.regularizationScaled.resize(getNumModels());  _opts.regularizationScaled.resize(getNumModels());
        _opts.planeOffset.resize(getNumModels());           _opts.planeOffset.back() = _opts.planeOffset.front();
        _opts.planeNormal.resize(getNumModels());           _opts.planeNormal.back() = _opts.planeNormal.front();
        _opts.lambdaIntersection.resize(getNumModels()*getNumModels());
        for (int i=(getNumModels()-1)*(getNumModels()-1); i<_opts.lambdaIntersection.size(); ++i) { _opts.lambdaIntersection[i] = 0; }

    }

    CheckCudaDieOnError();

    return true;
}


void TrackerNoObs::updateModel(const int modelNum,
                          const float modelSdfResolution,
                          const float modelSdfPadding) {

    int modelID = _mirroredModels[modelNum]->getModelID();
    delete _mirroredModels[modelNum];

    HostOnlyModel model;
    readModelXML(_filenames[modelNum].c_str(),model);

    for (std::map<std::string,float>::const_iterator it = _sizeParams[modelNum].begin();
         it != _sizeParams[modelNum].end(); ++it) {
        model.setSizeParam(it->first,it->second);
    }

    model.computeStructure();
    model.voxelize(modelSdfResolution,modelSdfPadding);
    MirroredModel * newModel = new MirroredModel(model,
                                                 make_uint3(0),
                                                 0,
                                                 make_float3(0),
                                                 modelID);
    _mirroredModels[modelNum] = newModel;

    _T_fms->hostPtr()[modelNum] = newModel->getDeviceTransformsModelToFrame(); _T_fms->syncHostToDevice();
    _sdfFrames->hostPtr()[modelNum] = newModel->getDeviceSdfFrames();          _sdfFrames->syncHostToDevice();
    _sdfs->hostPtr()[modelNum] = newModel->getDeviceSdfs();                    _sdfs->syncHostToDevice();

    newModel->setPose(_estimatedPoses[modelNum]);

    CheckCudaDieOnError();
}

float TrackerNoObs::optimizePoses() {

    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return -1.f;
    }

    float error = _optimizer->optimizePoses(_mirroredModels,
        _estimatedPoses,
        nullptr,
        nullptr,
        0,
        0,
        _opts,
        *_T_mcs,
        *_T_fms,
        *_sdfFrames,
        *_sdfs,
        *_nSdfs,
        *_distanceThresholds,
        *_normalThresholds,
        *_planeOffsets,
        *_planeNormals,
        _collisionClouds,
        _intersectionPotentialMatrices,
        _dampingMatrices,
        _priors);
    return error;
}

float TrackerNoObs::getError() {
    if (!initialized()) {
        std::cerr << "the tracker is not initialized properly for tracking. make sure a depth source and a model have been added" << std::endl;
        return 0.f;
    }

    float error = _optimizer->getError(_mirroredModels, _estimatedPoses, _opts,
        *_T_mcs, _collisionClouds, _intersectionPotentialMatrices, _priors);
    return error;
}

void TrackerNoObs::setIntersectionPotentialMatrix(const int modelNum, const int * mx) {
    delete _intersectionPotentialMatrices[modelNum];
    const int nSdfs = _mirroredModels[modelNum]->getNumSdfs();
    _intersectionPotentialMatrices[modelNum] = new MirroredVector<int>(nSdfs*nSdfs);
    memcpy(_intersectionPotentialMatrices[modelNum]->hostPtr(),mx,nSdfs*nSdfs*sizeof(int));
    _intersectionPotentialMatrices[modelNum]->syncHostToDevice();
}

std::vector<std::shared_ptr<ContactPrior> >
    TrackerNoObs::addContactPoints(const std::vector<float3> &contact_points,
    const int src_model_idx, const int dst_model_idx, const float weight) {
    std::vector<std::shared_ptr<ContactPrior> > priors;
    for (float3 contact_point : contact_points) {
        std::shared_ptr<ContactPrior> p =
            std::make_shared<ContactPrior>(src_model_idx, dst_model_idx, 0, -1, weight,
            contact_point, weight/2.f, false, true, true);
        addPrior(p);
        priors.push_back(p);
    }
    return priors;
}

void TrackerNoObs::addThumbContact(const float3 &point, const int src_sdf_idx,
    const int src_model_idx, const int dst_model_idx, const float weight) {
    std::shared_ptr<Prior> p = std::make_shared<ContactPrior>(src_model_idx,
        dst_model_idx, src_sdf_idx, 0, weight, point, 100.f, false, true, false);
}

}