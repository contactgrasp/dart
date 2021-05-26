#ifndef TRACKER_NO_OBS_H
#define TRACKER_NO_OBS_H

#include <vector>
#include <memory>

#include "model/mirrored_model.h"
#include "pose/pose.h"
#include "optimization/optimization.h"
#include "optimization/optimizer.h"
#include "optimization/priors.h"
#include "util/string_format.h"

namespace dart {

const int versionMajor = 0;
const int versionMinor = 0;
const int versionRevision = 1;

const inline std::string getVersionString() { return stringFormat("%d.%d.%d",versionMajor,versionMinor,versionRevision); }

class TrackerNoObs {

public:

    TrackerNoObs();
    ~TrackerNoObs();

    bool addModel(const std::string & filename,
                  const float modelSdfResolution = 0.005,
                  const float modelSdfPadding = 0.10,
                  PoseReduction * poseReduction = 0,
                  const float collisionCloudDensity = 1e5,
                  const bool cacheSdfs = true);

    void updateModel(const int modelNum,
                     const float modelSdfResolution = 0.005,
                     const float modelSdfPadding = 0.10);

    inline void updatePose(const int modelNum) {
        _estimatedPoses[modelNum].projectReducedToFull();
        _mirroredModels[modelNum]->setPose(_estimatedPoses[modelNum]);
    }

    /**
     * @brief This function runs the optimizer to infer the poses of all tracked models in the currently observed frame.
     * @param opts A struct setting up various optimzation parameters.
     */
    float optimizePoses();
    float getError();

    // accessors
    inline int getNumModels() const { return _mirroredModels.size(); }

    inline const MirroredModel & getModel(const int modelNum) const { return * _mirroredModels[modelNum]; }
    inline MirroredModel & getModel(const int modelNum) { return * _mirroredModels[modelNum]; }

    inline const Pose & getPose(const int modelNum) const { return _estimatedPoses[modelNum]; }
    inline Pose & getPose(const int modelNum) { return _estimatedPoses[modelNum]; }
    inline std::map<std::string,float> & getSizeParams(const int modelNum) { return _sizeParams[modelNum]; }

    inline const float4 * getCollisionCloud(const int modelNum) const { return _collisionClouds[modelNum]->hostPtr(); }
    inline const float4 * getDeviceCollisionCloud(const int modelNum) const { return _collisionClouds[modelNum]->devicePtr(); }
    inline int getCollisionCloudSize(const int modelNum) const { return _collisionClouds[modelNum]->length(); }
    inline int getCollisionCloudSdfStart(const int modelNum, const int sdfNum) const { return _collisionCloudSdfStarts[modelNum][sdfNum]; }
    inline int getCollisionCloudSdfLength(const int modelNum, const int sdfNum) const { return _collisionCloudSdfLengths[modelNum][sdfNum]; }

    void setIntersectionPotentialMatrix(const int modelNum, const int * mx);
    const int * getIntersectionPotentialMatrix(const int modelNum) const { return _intersectionPotentialMatrices[modelNum]->hostPtr(); }

    const OptimizerNoObs * getOptimizer() const { return _optimizer; }

    Eigen::MatrixXf & getDampingMatrix(const int modelNum) { return *_dampingMatrices[modelNum]; }

    OptimizationOptions & getOptions() { return _opts; }

//protected:
    OptimizerNoObs * getOptimizer() { return _optimizer; }

    void addPrior(const std::shared_ptr<Prior> &prior) { _priors.push_back(prior); }
    void clearPriors() {_priors.clear(); }

    std::vector<std::shared_ptr<ContactPrior> >
        addContactPoints(const std::vector<float3> &points,
        const int src_model_idx=1, const int dst_model_idx=0, const float weight=1.f);
    void addThumbContact(const float3 &point, const int src_sdf_idx,
        const int src_model_idx=0, const int dst_model_idx=1, const float weight=5.f);

private:

    inline bool initialized() { return _optimizer != 0 && _estimatedPoses.size() > 0; }

    OptimizerNoObs * _optimizer;
    OptimizationOptions _opts;

    std::vector<MirroredModel *> _mirroredModels;
    std::vector<std::map<std::string,float> > _sizeParams;
    std::vector<PoseReduction *> _ownedPoseReductions;
    std::vector<Pose> _estimatedPoses;

    std::vector<std::string> _filenames;

    std::vector<Eigen::MatrixXf *> _dampingMatrices;
    std::vector<std::shared_ptr<Prior> > _priors;

    // collision stuff
    std::vector<MirroredVector<float4> *> _collisionClouds;
    std::vector<std::vector<int> > _collisionCloudSdfStarts;
    std::vector<std::vector<int> > _collisionCloudSdfLengths;
    std::vector<MirroredVector<int> *> _intersectionPotentialMatrices;

    // scratch space
    MirroredVector<SE3> * _T_mcs;
    MirroredVector<SE3 *> * _T_fms;
    MirroredVector<int *> * _sdfFrames;
    MirroredVector<const Grid3D<float> *> * _sdfs;
    MirroredVector<int> * _nSdfs;
    MirroredVector<float> * _distanceThresholds;
    MirroredVector<float> * _normalThresholds;
    MirroredVector<float> * _planeOffsets;
    MirroredVector<float3> * _planeNormals;
};

}

#endif // TRACKER_NO_OBS_H