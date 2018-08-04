#include "SoftmaxRegression.h"

// DivLogLikelihood_Kernel.cu

/**
 * Performs dLogL on classed data with 2 features, 
 * populating vectors of derivative terms for summation later
 * in the algorithm. This function should be called once for each 
 * class of points, asynchronously if you have the device memory for it.
 * @param points        A device pointer to the float2 data
 * @param dxTerms       A device pointer to be populated with derivative terms corresponding to the first feature "x"
 * @param dyTerms       A device pointer to be populated with derivative terms corresponding to the first feature "y"
 * @param devWeights    A device pointer to the parameter weights (Theta's) at the current iteration, float2 x numClasses
 * @param numPoints     The total number of points being classified, for bounds checking with maximum thread index
 * @param classNum      The class number for which the derivative is being calculated.
 * @param numClasses    THe total number of classes in the model
 **/

__global__ void divLogLikelihood(const float2* __restrict__ points,
                                 float*        dxTerms,
                                 float*        dyTerms,
                                 float2*       __restrict__ devWeights,
                                 const int     numPoints,                   // number of points in a class                              (m / numClasses)
                                 const int     classNum,                    // the index of the class of the weight being operated on   (k)
                                 const int     numClasses)                  // the total number of classes                              (K)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;                   // the index of the point                                   (i)
    if (pointIdx >= numPoints * numClasses) { return; }     // out of array bounds check

        // determine class of the point. I didn't take up data for this 
        // the points are just sorted by class in blocks of length #numPoints
        // Obviously a real data set could be sorted and bookmarked to achieve this.

    int pointClass = static_cast<int>(pointIdx / numPoints);
    float indicatorTerm = 0.0;
    if (pointClass == classNum)
    {
        indicatorTerm = 1.0;
    }

    float2 point = points[pointIdx];
    float expxTerm = 0.0;
    float expyTerm = 0.0;
    float sumxTerm = 0.0;
    float sumyTerm = 0.0;

    expxTerm = expf(devWeights[classNum].x * point.x);
    for (int classIdx = 0; classIdx < numClasses; ++classIdx)
    {
        sumxTerm += expf(devWeights[classIdx].x * point.x);
    }
    float testx = point.x * (indicatorTerm - (expxTerm / (1.0 + sumxTerm)));
    dxTerms[pointIdx] = testx;

    expyTerm = expf(devWeights[classNum].y * point.y);
    for (int classIdx = 0; classIdx < numClasses; ++classIdx)
    {
        sumyTerm += expf(devWeights[classIdx].y * point.y);
    }
    float testy = point.y * (indicatorTerm - (expyTerm / (1.0 + sumyTerm)));
    dyTerms[pointIdx] = testy;   
}