// local includes 
#include "GLInstance.h"
#include "SoftmaxInitializer.h"
#include "SoftmaxRegression.h"

int main()
{
	auto settings 	 = std::make_shared<SoftmaxSettings>();
	auto data	 	 = std::make_shared<SoftmaxData>();
	auto initializer = std::make_shared<SoftmaxInitializer>();

	initializer->initialize("config", settings, data);
	gpuErrchk(cudaDeviceSynchronize());
	auto glInstance  = std::make_shared<GLInstance>(settings);
	auto regression  = std::make_shared<SoftmaxRegression>(settings, data, glInstance);

	regression->execute();

	return 0;
}