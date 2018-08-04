#include "SoftmaxRegression.h"

__global__ void Color(float3* __restrict__ colors,
					  float* __restrict__ probField,
					  float3* __restrict__ rawColorMap,
					  const float2* __restrict__ points,
					  unsigned int simWidth,
					  unsigned int simHeight,
					  float mapMin,
					  float mapMax) // TODO
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	int quadIdx = x + simWidth*y;

	if (quadIdx >= (simWidth * simHeight))
	{
		printf("BAD: Out of colors[] boundary in Color<>, %d , max: %d\n", quadIdx*4, simWidth*simHeight*4); 
		return;
	}

	int mapped = (int)(0 + (((probField[quadIdx] - mapMin) * (511 - 0)) / (mapMax - mapMin)));

	for(int i = 0; i < 4; i++)
	{

		if(mapped >= 511)
		{
			colors[4*quadIdx+i] = rawColorMap[510];
		}
		else if(probField[quadIdx] <= 0)
		{
			colors[4*quadIdx+i] = make_float3(0.15,0.15,0.16);
		}
		else
		{
			colors[4*quadIdx+i] = rawColorMap[mapped];
		}
	}

}

__global__ void assignPointQuadIndices(const float2* __restrict__ points, 
									   int* quadIndices,
									   const int numPoints,                       // number of points in a class (m / numClasses)
                                	   const int numClasses,                      // the total number of classes (K)
									   unsigned int simWidth,
									   unsigned int simHeight)
{
    unsigned int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;          // the index of the point (i)
    if (pointIdx >= numPoints * numClasses) return;
    int quadX = (int)((1.0 + points[pointIdx].x)*((float)simWidth/2.0));
    int quadY = (int)((1.0 - points[pointIdx].y)*((float)simHeight/2.0));
    if (pointIdx == 0)
    {
    	printf("Assigning point quad x, y, quadidx: %d, %d, %d\n", quadX, quadY, quadX+simWidth*quadY);
    	printf("Assigning point quad x, y, quadidx: %f, %f, %d\n", points[pointIdx].x, (1.0 / (simWidth/2)), simWidth);
    }
    if ((quadX + simWidth * quadY) >= simWidth * simHeight)
    {
    	return;
    }
    if ((quadX + simWidth * quadY) <= 0)
    {
    	printf("Got a negative quadidx\n");
    	return;
    }
    quadIndices[pointIdx] = quadX + simWidth * quadY;
    __syncthreads();
}

__global__ void ColorPointQuads(float3* colors, 
								const int* __restrict__ quadIndices,
                                const int numPoints,                       // number of points in a class (m / numClasses)
                                const int numClasses,
                                unsigned int windowWidth,
                                unsigned int windowHeight)                      // the total number of classes (K)
{
	unsigned int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints * numClasses) return;
    int quadIndex = quadIndices[pointIdx];
    if (quadIndex >= windowWidth * windowHeight - 1) return;
    if (quadIndex <= 0) return;
	for (int i = 0; i < 4; i++)
	{
		colors[4 * quadIndex + i] = make_float3(0.1, 1.0, 0.2);
	}

	__syncthreads();
}