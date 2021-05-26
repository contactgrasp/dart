#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "optimization.h"
#include "prediction_renderer.h"
#include "priors.h"
#include "depth_sources/depth_source.h"
#include "model/mirrored_model.h"
#include "pose/pose.h"
#include <memory>

namespace dart {

struct Observation {
    const float4 * dVertMap, * dNormMap;
    const int width, height;
    Observation(const float4 * dVertMap_, const float4 * dNormMap_, const int width_, const int height_) :
        dVertMap(dVertMap_), dNormMap(dNormMap_), width(width_), height(height_) { }
};


class OptimizerNoObs {
 public:
  OptimizerNoObs();
  virtual ~OptimizerNoObs();
  virtual void optimizePose(MirroredModel & model,
      Pose & pose,
      const float4 * dObsVertMap,
      const float4 * dObsNormMap,
      const int width,
      const int height,
      OptimizationOptions & opts,
      MirroredVector<float4> & collisionCloud,
      MirroredVector<int> & intersectionPotentialMatrix,
      const PosePrior * prior = 0) {}

  virtual float optimizePoses(std::vector<MirroredModel *> & models,
                   std::vector<Pose> & poses,
                   const float4 * dObsVertMap,
                   const float4 * dObsNormMap,
                   const int width,
                   const int height,
                   OptimizationOptions & opts,
                   MirroredVector<SE3> & T_mcs,
                   MirroredVector<SE3 *> & T_fms,
                   MirroredVector<int *> & sdfFrames,
                   MirroredVector<const Grid3D<float> *> & sdfs,
                   MirroredVector<int> & nSdfs,
                   MirroredVector<float> & distanceThresholds,
                   MirroredVector<float> & normalThresholds,
                   MirroredVector<float> & planeOffsets,
                   MirroredVector<float3> & planeNormals,
                   std::vector<MirroredVector<float4> *> & collisionClouds,
                   std::vector<MirroredVector<int> *> & intersectionPotentialMatrices,
                   std::vector<Eigen::MatrixXf *> & dampingMatrices,
                   std::vector<std::shared_ptr<Prior> > & priors);

  void computeJacobiansAndError(std::vector<MirroredModel *> &models,
      std::vector<Pose> &poses,
      OptimizationOptions &opts,
      std::vector<MirroredVector<float4> *> &collisionClouds,
      std::vector<MirroredVector<int> *> &intersectionPotentialMatrices,
      std::vector<Eigen::MatrixXf *> &dampingMatrices,
      std::vector<std::shared_ptr<Prior> > &priors,
      std::vector<int> modelOffsets,
      std::vector<int> priorOffsets,
      Eigen::SparseMatrix<float> &sparseJTJ,
      Eigen::VectorXf &fullJTe,
      float &error);

  bool shouldStop(OptimizationOptions &opts, const int &n_iters);

  virtual float getError(std::vector<MirroredModel *> & models,
                   std::vector<Pose> & poses,
                   OptimizationOptions & opts,
                   MirroredVector<SE3> & T_mcs,
                   std::vector<MirroredVector<float4> *> & collisionClouds,
                   std::vector<MirroredVector<int> *> & intersectionPotentialMatrices,
                   std::vector<std::shared_ptr<Prior> > & priors);

  const float * getDeviceDebugIntersectionError() const { return _dDebugIntersectionError; }

  const Eigen::MatrixXf * getJTJ(const int modelNum) const { return _JTJ[modelNum]; }
  uchar3 * getJTJimg() const { return _JTJimg->hostPtr(); }

 protected:

  void init();

  void unpack(Eigen::VectorXf & eJ,
      Eigen::MatrixXf & JTJ,
      float & e,
      float * sys,
      const float multiplier,
      const int dimensions);

  void computeSelfIntersectionContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                           const MirroredModel & model, const Pose & pose,
                                           const OptimizationOptions & opts,
                                           const MirroredVector<float4> & collisionCloud,
                                           const MirroredVector<int> & intersectionPotentialMatrix,
                                           const int nModels, const int debugOffset);

  void computeIntersectionContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                       const MirroredModel & srcModel, const MirroredModel & dstModel,
                                       const Pose & pose, const OptimizationOptions & opts,
                                       const MirroredVector<float4> & collisionCloud,
                                       const int nModels, const int debugOffset);

  float * _dDebugIntersectionError;

  float * _dError;
  int _maxModels;
  int _maxDims;
  int _maxIntersectionSites;

  MirroredVector<float> * _result;
  std::vector<Eigen::MatrixXf *> _JTJ;
  MirroredVector<uchar3> * _JTJimg;
};

class Optimizer: public OptimizerNoObs {
public:
    Optimizer(const DepthSourceBase * depthSource,
              const int predictionWidth = -1, const int predictionHeight = -1);
    Optimizer(const int depthWidth, const int depthHeight, const float2 focalLength,
              const int predictionWidth = -1, const int predictionHeight = -1);
    ~Optimizer();

    void optimizePose(MirroredModel & model,
                      Pose & pose,
                      const float4 * dObsVertMap,
                      const float4 * dObsNormMap,
                      const int width,
                      const int height,
                      OptimizationOptions & opts,
                      MirroredVector<float4> & collisionCloud,
                      MirroredVector<int> & intersectionPotentialMatrix,
                      const PosePrior * prior = 0);

    float optimizePoses(std::vector<MirroredModel *> & models,
                       std::vector<Pose> & poses,
                       const float4 * dObsVertMap,
                       const float4 * dObsNormMap,
                       const int width,
                       const int height,
                       OptimizationOptions & opts,
                       MirroredVector<SE3> & T_mcs,
                       MirroredVector<SE3 *> & T_fms,
                       MirroredVector<int *> & sdfFrames,
                       MirroredVector<const Grid3D<float> *> & sdfs,
                       MirroredVector<int> & nSdfs,
                       MirroredVector<float> & distanceThresholds,
                       MirroredVector<float> & normalThresholds,
                       MirroredVector<float> & planeOffsets,
                       MirroredVector<float3> & planeNormals,
                       std::vector<MirroredVector<float4> *> & collisionClouds,
                       std::vector<MirroredVector<int> *> & intersectionPotentialMatrices,
                       std::vector<Eigen::MatrixXf *> & dampingMatrices,
                       std::vector<std::shared_ptr<Prior> > & priors);

    const float4 * getDevicePredictedPoints() const { return _predictionRenderer->getDevicePrediction(); }

    /**
     * @brief Gets the pointer to the observation to model error map. This map has a 2D structure with a width and height equal to that of the observed depth map. The memory pointed to is only updated when Tracker::optimizePoses is called with OptimizationOptions::debugObsToModErr set to true.
     * @return A pointer to device memory storing the observation to model error map.
     */
    const float * getDeviceDebugErrorObsToMod() const { return _dDebugObsToModError; }

    /**
     * @brief Gets the pointer to the model to observation error map. This map has a 2D structure with a width and height equal to that of the predicted depth map. The memory pointed to is only updated when Tracker::optimizePoses is called with OptimizationOptions::debugModToObsErr set to true.
     * @return A pointer to device memory storing the model to observation error map.
     */
    const float * getDeviceDebugErrorModToObs() const { return _dDebugModToObsError; }

    /**
     * @brief Gets the pointer to the observation to model data association map. This map has a 2D structure with a width and height equal to that of the observed depth map. The memory pointed to is only updated when Tracker::optimizePoses is called with OptimizationOptions::debugObsToModDA set to true.
     * @return A pointer to device memory storing the observation to model data association map.
     */
    const int * getDeviceDebugDataAssociationObsToMod() const { return _dDebugDataAssocObsToMod; }

    /**
     * @brief Gets the pointer to the model to observation data association map. This map has a 2D structure with a width and height equal to that of the predicted depth map. The memory pointed to is only updated when Tracker::optimizePoses is called with OptimizationOptions::debugModToObsDA set to true.
     * @return A pointer to device memory storing the model to observation data association map.
     */
    const int * getDeviceDebugDataAssociationModToObs() const { return _dDebugDataAssocModToObs; }

    const float4 * getDeviceDebugPredictedNormMap() const { return _dDebugObsToModNorm; }
    const float4 * getDeviceDebugObservedNormMap() const { return _dDebugModToObsNorm; }
    int getPredictionWidth() const { return _predictionRenderer->getWidth(); }
    int getPredictionHeight() const { return _predictionRenderer->getHeight(); }

    const float * getSoftDataAssociation() const { return _associationWeights->hostPtr(); }

    inline float getErrObsToMod(const int modelNum, const int iteration)       const { return _iterationSummaries[modelNum][iteration].errObsToMod; }
    inline float getErrModToObs(const int modelNum, const int iteration)       const { return _iterationSummaries[modelNum][iteration].errModToObs; }
    inline int getNumAssociatedPoints(const int modelNum, const int iteration) const { return _iterationSummaries[modelNum][iteration].nAssociatedPoints; }
    inline int getNumPredictedPoints(const int modelNum, const int iteration)  const { return _iterationSummaries[modelNum][iteration].nPredictedPoints; }
    inline float getErrPerObsPoint(const int modelNum, const int iteration)    const { return getErrObsToMod(modelNum,iteration) / getNumAssociatedPoints(modelNum,iteration); }
    inline float getErrPerModPoint(const int modelNum, const int iteration)    const { return getErrModToObs(modelNum,iteration) / getNumPredictedPoints(modelNum,iteration); }

    const unsigned char * getDebugBoxIntersections() const { return _predictionRenderer->getDebugBoxIntersection(); }

    void computePredictedPointCloud(const std::vector<MirroredModel*> & models) {

        std::vector<const MirroredModel *> constModelPtrs(models.size());
        for (int m=0; m<models.size(); ++m) {
            constModelPtrs[m] = models[m];
        }
        _predictionRenderer->raytracePrediction(constModelPtrs,_depthPredictStream);
    }

    void debugPredictionRay(const std::vector<MirroredModel*> & models, const int x, const int y,
                            std::vector<MirroredVector<float3> > & boxIntersects, std::vector<MirroredVector<float2> > & raySteps) {
        std::vector<const MirroredModel *> constModelPtrs(models.size());
        for (int m=0; m<models.size(); ++m) {
            constModelPtrs[m] = models[m];
        }
        _predictionRenderer->debugPredictionRay(constModelPtrs,x,y,boxIntersects,raySteps);
    }

private:

    void init(const int depthWidth, const int deptHheight, const float2 focalLength,
              const int predictionWidth, const int predictionHeight);

    void generateObsSdf(MirroredModel & model,
                        const Observation & observation,
                        const OptimizationOptions & opts);

    void generateObsSdfSplatZeroAndDistanceTransform(MirroredModel & model,
                                                     const Observation & observation,
                                                     const OptimizationOptions & opts);

    void generateObsSdfDirectTruncated(MirroredModel & model,
                                       const Observation & observation,
                                       const OptimizationOptions & opts);

    void computeObsToModContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                     const MirroredModel & model, const Pose & pose,
                                     const OptimizationOptions & opts, const Observation & observation);

    void computeModToObsContribution(Eigen::VectorXf & eJ, Eigen::MatrixXf & JTJ, float & error,
                                     const MirroredModel & model, const Pose & pose, const SE3 & T_obsSdf_c,
                                     const OptimizationOptions & opts);

    PredictionRenderer * _predictionRenderer;

    // GPU memory
    int * _dDebugDataAssocObsToMod;
    int * _dDebugDataAssocModToObs;
    float * _dDebugObsToModError;
    float * _dDebugModToObsError;
    float4 * _dDebugModToObsNorm;
    float4 * _dDebugObsToModNorm;
    float4 * _dDebugObsToModJs;

    MirroredVector<int> * _lastElements;
    MirroredVector<DataAssociatedPoint *> * _dPts;

    cudaStream_t _depthPredictStream;
    cudaStream_t _posInfoStream;

    struct IterationSummary {
        float errObsToMod, errModToObs;
        int nAssociatedPoints, nPredictedPoints;
    };

    std::vector<std::vector<IterationSummary> > _iterationSummaries;

    // EM stuff
    MirroredVector<float> * _mixWeights;
    MirroredVector<float> * _sigmas;
    MirroredVector<float> * _associationWeights;
};

}

#endif // OPTIMIZER_H
