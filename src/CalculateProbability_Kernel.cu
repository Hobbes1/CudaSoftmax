#include "SoftmaxRegression.h"

	// With the set of weights at a given step, calculate the probability field 
	// at each of the quad (pixel) positions, for visualization

__global__ void CalculateProbability (float* __restrict__ probability,
									  float2* __restrict__ weights,
									  int plotNum,
									  int numClasses,
									  unsigned int windowWidth,
					  				  unsigned int windowHeight)
{
	int quadX = blockIdx.x * blockDim.x + threadIdx.x;
	int quadY = blockIdx.y * blockDim.y + threadIdx.y;
	int quadIdx = quadX + windowWidth * quadY;
	if (plotNum >= numClasses)
	{
		return;
	}
	// given the window dimensions, and assuming it's scaled as a 2x2 box centered on (0,0)
	float quadXpos = -1.0 + 2.0 * (float)quadX / (float)windowWidth;
	float quadYpos = 1.0 - 2.0 * (float)quadY / (float)windowHeight;

	float pRes = 0.0;
	float expTerm = 0.0;
	float sumTerm = 0.0;

	if (plotNum == -1)
	{
		for (int classIdx = 0; classIdx < numClasses; classIdx++)
		{
	        expTerm = expf(weights[classIdx].x * quadXpos + weights[classIdx].y * quadYpos);
	        for (int classIdx2 = 0; classIdx2 < numClasses; ++classIdx2)
	        {
	            sumTerm += expf(weights[classIdx2].x * quadXpos + weights[classIdx2].y * quadYpos);
	        }
			pRes += expTerm / (sumTerm);
			//sumTerm = 0.0;
		}
	}
	else 
	{
		expTerm = expf(weights[plotNum].x * quadXpos + weights[plotNum].y * quadYpos);
        for (int classIdx = 0; classIdx < numClasses; ++classIdx)
        {
            sumTerm += expf(weights[classIdx].x * quadXpos + weights[classIdx].y * quadYpos);
        }
		pRes += expTerm / (sumTerm);
	}

	probability[quadIdx] = pRes;
}

__global__ void RemoveLowers (float* __restrict__ probability,
							  unsigned int windowWidth,
							  unsigned int windowHeight,
			  				  float mapMin,
			  				  float mapMax)
{
	int quadX = blockIdx.x * blockDim.x + threadIdx.x;
	int quadY = blockIdx.y * blockDim.y + threadIdx.y;
	int quadIdx = quadX + windowWidth * quadY;

	if (probability[quadIdx] < 0.5)
	{
		probability[quadIdx] = 0.0;
	}

}