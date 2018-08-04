#ifndef SOFTMAXINITIALIZER_H
#define SOFTMAXINITIALIZER_H
// Thrust includes
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

// CUDA includes
#include <vector_types.h>
#include <cuda_profiler_api.h>

// std includes
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <random>
#include <memory>
#include <string>

// Note that you could template the data such that float2
// is replaced by floatN where N = number of features.
// Not the colors of course, unless you have a really neat screen.

struct SoftmaxData
{
	thrust::host_vector<float2> hostPoints;				// point data classed by index ranges
	thrust::device_vector<float2> devPoints;
	float2* devPointsPtr;

	thrust::host_vector<float2> hostWeights;			// weight set for 2-feature data set.
	thrust::device_vector<float2> devWeights;
	float2* devWeightsPtr;

	thrust::host_vector<float2> hostAlphas;				// alpha parameters. Does every weight need one? Sure why not.
		
	thrust::host_vector<float3> hostColorMap;			// color data for visualization
	thrust::device_vector<float3> devColorMap;
	float3* devColorMapPtr;

	thrust::host_vector<int> hostQuadIndices;			// quad indices where points lay, for visualization
	thrust::device_vector<int> devQuadIndices;
	int* devQuadIndicesPtr;

	std::vector<thrust::host_vector<float>> hostProbFields;			// probability field to populate and plot with
	std::vector<thrust::device_vector<float>> devProbFields;
	std::vector<float*> devProbFieldPtrs;

	// class 		feature 			pointIdx
	std::vector<std::vector<thrust::host_vector<float>>> 	hostDivLogLTerms;
	std::vector<std::vector<thrust::device_vector<float>>> 	devDivLogLTerms;
	std::vector<std::vector<float*>> 						devDivLogLPtrs;

	thrust::host_vector<uint8_t> hostPixelData;	// pixel data to convert and xfer to host for recordings
	thrust::device_vector<uint8_t> devPixelData;
	uint8_t* devPixelDataPtr;
};

struct SoftmaxSettings
{
	int numClasses;
	int numPoints;
	int numFeatures;
	int plotNum;
	float stdDev;
	float* xAlphas;
	float* yAlphas;
	int windowHeight;
	int windowWidth;
	bool recording;
	int frames;
	std::string colorMap = "";
};

	// Takes command line arguments to populate a settings instance 
    // and a data instance for use by openGL and the regression kernels.
class SoftmaxInitializer
{
	public:

		SoftmaxInitializer();

		~SoftmaxInitializer();

			// Initialize a data and settings instance

		void initialize(std::string aConfigFileName,
						std::shared_ptr<SoftmaxSettings>& aSettings,
					    std::shared_ptr<SoftmaxData>& aData);

	private:

		void _loadSettings(std::string aConfigFileName,
				      	   std::shared_ptr<SoftmaxSettings>& aSettings);

		void _loadData(std::shared_ptr<SoftmaxSettings>& aSettings,
		      		   std::shared_ptr<SoftmaxData>& aData);

		void _genGaussians(std::shared_ptr<SoftmaxSettings>& aSettings,
		  		  		   thrust::host_vector<float2>& hostPoints);
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif 