#include "SoftmaxRegression.h"
__global__ void divLogLikelihood(const float2* __restrict__ points,
                                 float*         dxTerms,
                                 float*         dyTerms,
                                 float2*       __restrict__ devWeights,
                                 const int numPoints,                       // number of points in a class                              (m / numClasses)
                                 const int classNum,                        // the index of the class of the weight being operated on   (k)
                                 const int numClasses)                      // the total number of classes                              (K)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;                   // the index of the point                                   (i)
    if (pointIdx >= numPoints * numClasses || pointIdx < 0)
    {
        return;
    }

        // determine class of the point. I didn't take up data for this 
        // the points are just sorted by class in blocks of length numPoints;

    int pointClass = static_cast<int>(pointIdx / numPoints);

    float2 point = points[pointIdx];
    float expxTerm = 0.0;
    float expyTerm = 0.0;
    float sumxTerm = 0.0;
    float sumyTerm = 0.0;
    float indicatorTerm = 0.0;
    if (pointClass == classNum)
    {
        indicatorTerm = 1.0;
    }

    expxTerm = expf(devWeights[pointClass].x * point.x);
    for (int classIdx = 0; classIdx < numClasses; ++classIdx)
    {
        sumxTerm += expf(devWeights[classIdx].x * point.x);
    }
    float testx = point.x * (indicatorTerm - (expxTerm / (1.0 + sumxTerm)));

    if (!isnan(testx)) 
    {
        dxTerms[pointIdx] = testx;
    }
    else 
       printf("Got a nan at classLoop: %d pointIdx: %i... point.x:%f indicator: %f expTerm: %f sumTerm: %f \n", classNum, pointIdx, point.x, indicatorTerm, expxTerm, sumxTerm);

    __syncthreads();
    expyTerm = expf(devWeights[pointClass].y * point.y);

    for (int classIdx = 0; classIdx < numClasses; ++classIdx)
    {
        sumyTerm += expf(devWeights[classIdx].y * point.y);
    }
    float testy = point.y * (indicatorTerm - (expyTerm / (1.0 + sumyTerm)));

    if (!isnan(testy))
    {
        dyTerms[pointIdx] = testy;
    }
    else 
       printf("Got a nan at classLoop: %d pointIdx: %i... point.y:%f indicator: %f expTerm: %f sumTerm: %f \n", classNum, pointIdx, point.y, indicatorTerm, expyTerm, sumyTerm);
    
}