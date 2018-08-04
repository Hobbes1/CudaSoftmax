#ifndef SOFTMAXREGRESSION_H
#define SOFTMAXREGRESSION_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
#define CUDA_KERNEL_MEMBER __global__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_KERNEL_MEMBER
#endif

// project includes
#include "SoftmaxInitializer.h"
#include "GLInstance.h"
#include "WritePNG.h"

#include <memory>

#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/inner_product.h>
#include <cmath>

#include <GL/glut.h>

	// class takes data pointers and runs regression CUDA kernels. 
// extern class GLInstance;

class SoftmaxRegression
{

public:

	SoftmaxRegression(std::shared_ptr<SoftmaxSettings> aSettings,
				      std::shared_ptr<SoftmaxData> aData,
				      std::shared_ptr<GLInstance> aGLInstance);

	~SoftmaxRegression();

	void execute();


private:

	std::shared_ptr<SoftmaxSettings> mSettings;

	std::shared_ptr<SoftmaxData> mData;

    std::shared_ptr<GLInstance> mGL;

		// Set any additional class variables that the kernels will require
	    // based on the settings and data members. In particular, thread 
		// dimensions for the various kernels. 

	void initialize();

		// passthrough to GLInstance::draw()

	void Draw();

		// Thread dimension member variables set by initialize

	int mRegressionBlocks;

	int mRegressionTPB;

	dim3 mColorTPB;

	dim3 mColorBlocks;	

};
	/* If a kernel relies on __shared__ data that has been initialized 
	 * at some time (_initialize()) it needs to include as a fourth parameter
	 * the size of the data, as in:
	 * kernel_name<<<blocks, threads_per_block, shared_mem_size_in_bytes...>>>(...);
	 */

	/* Weights for each "Jth" feature; x, y, z
	 * and for each "Kth" class; 1 through 8
	 * These are constant for the duration of a kernel 
	 * call and set with cudaMemcpyToSymbol
	 */
extern __shared__ float2 devWeights[];

	/* Alpha regression rates for each "Jth" feature; x, y, z
	 * and for each "Kth" class; 1 through 8
	 */
//__constant__ float3 alphas[];


	/* Compute the Mth term of the summation of the derivative 
	 * of log likelihood for a particular parameter weight, in parallel
	 * This kernel will need to run for each parameter weight, and
	 * the output will need to be summed and scaled by the "learning" 
	 * rate before being used to modify a parameter weight.
	 */ 
CUDA_KERNEL_MEMBER void divLogLikelihood(const float2* __restrict__ points,
									 	 float*        __restrict__ dxTerms,
									 	 float* 	   __restrict__ dyTerm,
									 	 float2* 	   __restrict__ weights,
									 	 const int numPoints,						// number of points in a class                              (m / numClasses)
									 	 const int classNum,						// the index of the class of the weight being operated on   (k)
								 		 const int totalClassNum);					// the total number of classes

CUDA_KERNEL_MEMBER void Color(float3* colors,
							  float* probField,
						  	  float3* rawColorMap,
							  const float2* __restrict__ points,
							  unsigned int simWidth,
							  unsigned int simHeight,
							  float mapMin,
							  float mapMax);	

CUDA_KERNEL_MEMBER void assignPointQuadIndices(const float2* __restrict__ points, 
											   int* quadIndices,
											   const int numPoints,                       // number of points in a class (m / numClasses)
		                                	   const int numClasses,                      // the total number of classes (K)
											   unsigned int simWidth,
											   unsigned int simHeight);

CUDA_KERNEL_MEMBER void ColorPointQuads(float3* colors, 
										const int* __restrict__ quadIndices,
		                                const int numPoints,                       // number of points in a class (m / numClasses)
		                                const int numClasses,
		                                unsigned int windowWidth,
		                                unsigned int windowHeight);                     // the total number of classes (K)

CUDA_KERNEL_MEMBER void CalculateProbability (float* __restrict__ probability,
											  float2* __restrict__ weights,
		  									  int plotNum,	
											  int numClasses,
											  unsigned int windowWidth,
							  				  unsigned int windowHeight);

CUDA_KERNEL_MEMBER void RemoveLowers	     (float* __restrict__ probability,
											  unsigned int windowWidth,
											  unsigned int windowHeight,
							  				  float mapMin,
							  				  float mapMax);

CUDA_KERNEL_MEMBER void FormPNGData 	 (float3* colors,
										  unsigned char* pixelData, 
										  unsigned int simWidth, 
										  unsigned int simHeight);

// 	/* Computes the inner product between two 3-vectors,
// 	 * the parameters go left to right
// 	 */
// CUDA_CALLABLE_MEMBER void innerProd(float3 vectorT,
// 									float3 vector);

#endif