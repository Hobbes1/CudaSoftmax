#include "SoftmaxRegression.h"
#include "iomanip"

SoftmaxRegression::SoftmaxRegression(std::shared_ptr<SoftmaxSettings> aSettings,
							         std::shared_ptr<SoftmaxData> aData,
							         std::shared_ptr<GLInstance> aGLInstance) :
	mSettings(aSettings),
	mData(aData),
	mGL(aGLInstance)
{
	initialize();
	assignPointQuadIndices<<<mRegressionBlocks, mRegressionTPB>>>
						  (mData->devPointsPtr, mData->devQuadIndicesPtr,
						   mSettings->numPoints, mSettings->numClasses,           
						   mSettings->windowWidth, mSettings->windowHeight);
}

SoftmaxRegression::~SoftmaxRegression()
{
	
}

void SoftmaxRegression::execute()
{  
	int steps = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float fpsTime = 0.0;
	float mapMin = 0;
	float mapMax = 0;

	std::vector<std::vector<float>>  dLogLSums(mSettings->numClasses, std::vector<float>(mSettings->numFeatures, 0.0));		// numClasses x numFeatures sum of divLogL terms to set on each regression step
	std::vector<float> probNorms;
	probNorms.resize(mSettings->numClasses);
	std::string frameName;

	while(!glfwWindowShouldClose(mGL->window))
	{
		cudaEventRecord(start, 0);

		float3 *colorsPtr;
		gpuErrchk(cudaGraphicsMapResources(1, &mGL->cudaColorResource, 0));

		size_t numBytes;
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)&colorsPtr, &numBytes,
													   *&mGL->cudaColorResource));


			// Take dLogL for each class, filling each derivative vector

		for (int classIdx = 0; classIdx < mSettings->numClasses; ++classIdx)
		{

			divLogLikelihood<<<mRegressionBlocks, mRegressionTPB>>> 
				(mData->devPointsPtr,
				 mData->devDivLogLPtrs[classIdx][0],	// x divs
    		     mData->devDivLogLPtrs[classIdx][1],	// y divs
    		     mData->devWeightsPtr,
    		     mSettings->numPoints,																																	 
	  		     classIdx,
	  		     mSettings->numClasses);
				
			gpuErrchk(cudaDeviceSynchronize());
		}

		gpuErrchk(cudaDeviceSynchronize());

			// Sum the derivative vectors

		for (int classIdx = 0; classIdx < mSettings->numClasses; ++classIdx)
		{
			for (int featureIdx = 0; featureIdx < mSettings->numFeatures; ++featureIdx)
			{
		        dLogLSums[classIdx][featureIdx] = thrust::reduce(mData->devDivLogLTerms[classIdx][featureIdx].begin(), 
		        									             mData->devDivLogLTerms[classIdx][featureIdx].end());						  				  		 
			}
		}

		gpuErrchk(cudaDeviceSynchronize());

			// reset the derivative vectors for the next iteration

		for (int classIdx = 0; classIdx < mSettings->numClasses; ++classIdx)
		{
			for (int f = 0; f < mSettings->numFeatures; ++f)
			{
				thrust::fill(mData->devDivLogLTerms[classIdx][f].begin(), 
					         mData->devDivLogLTerms[classIdx][f].end(),
					         0.0);
			}
		}	

			// update the weights using the sums and scaling factors

		for (int classIdx = 0; classIdx < mSettings->numClasses; ++classIdx)
		{
			mData->hostWeights[classIdx].x += mData->hostAlphas[classIdx].x * dLogLSums[classIdx][0];
			mData->hostWeights[classIdx].y += mData->hostAlphas[classIdx].y * dLogLSums[classIdx][1];
		}

			// copy the updated weights from the host to the device for the next iteration

		mData->devWeights = mData->hostWeights;
		mData->devWeightsPtr = thrust::raw_pointer_cast(mData->devWeights.data());

		///
		///	The rest is visualization code
		///

		gpuErrchk(cudaDeviceSynchronize());

		// Populate a probability field for each set of class weights. Because they vary in magnitude, 
		// normalize them and scale them to the same maximum (1.0) before combining to plot.
		for (int classIdx = 0; classIdx < mSettings->numClasses; ++classIdx)
		{
			CalculateProbability<<< mColorBlocks, mColorTPB >>> 
			(mData->devProbFieldPtrs[classIdx],
			 mData->devWeightsPtr,
			 classIdx,
			 mSettings->numClasses,
			 mSettings->windowWidth,
			 mSettings->windowHeight);

			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaPeekAtLastError());

			probNorms[classIdx] = std::sqrt(thrust::inner_product(mData->devProbFields[classIdx].begin(), 
						  						mData->devProbFields[classIdx].end(), 
					  							mData->devProbFields[classIdx].begin(), 0.0f));
			using namespace thrust::placeholders;
			thrust::transform(mData->devProbFields[classIdx].begin(), mData->devProbFields[classIdx].end(), mData->devProbFields[classIdx].begin(), _1 /= probNorms[classIdx]);
			auto minMaxPtrs = thrust::minmax_element(mData->devProbFields[classIdx].begin(), mData->devProbFields[classIdx].end());
			mapMax = *minMaxPtrs.second;
			mapMin = *minMaxPtrs.first;
			thrust::transform(mData->devProbFields[classIdx].begin(), mData->devProbFields[classIdx].end(), mData->devProbFields[classIdx].begin(), _1 /= mapMax);

			RemoveLowers<<< mColorBlocks, mColorTPB >>>	     
						(mData->devProbFieldPtrs[classIdx],
						 mSettings->windowWidth,
						 mSettings->windowHeight,
						 mapMin,
						 mapMax);

			gpuErrchk(cudaDeviceSynchronize());
			gpuErrchk(cudaPeekAtLastError());
		}

		// sum them in place in the zero'th field
		for (int classIdx = 1; classIdx < mSettings->numClasses; ++classIdx)
		{
			thrust::transform(mData->devProbFields[0].begin(), mData->devProbFields[0].end(), mData->devProbFields[classIdx].begin(), mData->devProbFields[0].begin(), thrust::plus<float>());
			gpuErrchk(cudaDeviceSynchronize());
		}
		

		auto minMaxPtrs = thrust::minmax_element(mData->devProbFields[0].begin(), mData->devProbFields[0].end());
		mapMin = *minMaxPtrs.first;
		mapMax = *minMaxPtrs.second;
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		Color<<< mColorBlocks, mColorTPB >>> 	
			(colorsPtr,
			 mData->devProbFieldPtrs[0],
			 mData->devColorMapPtr,
			 mData->devPointsPtr,
			 mSettings->windowWidth, 
			 mSettings->windowHeight,
			 mapMin,
			 mapMax);
			
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		ColorPointQuads<<< mRegressionBlocks, mRegressionTPB >>> 
			(colorsPtr, 
			 mData->devQuadIndicesPtr,
             mSettings->numPoints,                    
             mSettings->numClasses,
             mSettings->windowWidth,
             mSettings->windowHeight);

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaGraphicsUnmapResources(1, &mGL->cudaColorResource, 0));

		Draw();

		if (mSettings->recording && steps < mSettings->frames)
		{
			frameName = FrameNameGen(steps, mSettings->frames);
			FormPNGData<<< mColorBlocks, mColorTPB >>> 		(colorsPtr, 
															 mData->devPixelDataPtr, 
															 mSettings->windowWidth, 
															 mSettings->windowHeight);

			gpuErrchk(cudaPeekAtLastError());

			gpuErrchk(cudaMemcpy(mData->hostPixelData.data(),
					   mData->devPixelDataPtr,
					   mSettings->windowWidth * mSettings->windowHeight * 3 * sizeof(unsigned char),
					   cudaMemcpyDeviceToHost));

			gpuErrchk(cudaPeekAtLastError());
			WritePNG(mData->hostPixelData.data(),
					 frameName,
					 mSettings->windowWidth,
					 mSettings->windowHeight);
		}

		steps++;
		cudaEventRecord(stop, 0);
		cudaEventElapsedTime(&fpsTime, start, stop); 
		char title[512];
		sprintf(title, "Cuda Softmax Regression: %12.2f fps, point count: %u, steps taken: %d", 1.0f/(fpsTime/1000.0f), mSettings->numPoints*mSettings->numClasses, steps);
		glfwSetWindowTitle(mGL->window, title);
		//|| steps == 1
		if(glfwGetKey(mGL->window, GLFW_KEY_ESCAPE)) {
			glfwSetWindowShouldClose(mGL->window, 1);
			std::cout << "Window closed" << std::endl;
		}
	}
	return;
}

void SoftmaxRegression::Draw()
{
	mGL->Draw();
}

void SoftmaxRegression::initialize()
{
		// Essentially attempt to have 
		// 256 threads per block as per occupation optimization. Though in all
		// honesty I have never benchmarked a thing.
		// Also hard coded for my device (3.5) specs which is a no-no

	int totalPoints = mSettings->numPoints * mSettings->numClasses;
	if (totalPoints <= 2 << 7)
	{
		mRegressionBlocks = 1;
	}
	else if (totalPoints <= 2 << 9)
	{
		mRegressionBlocks = 4;
	}
	else if (totalPoints <= 2 << 11)
	{
		mRegressionBlocks = 16;
	}
	else if (totalPoints <= 2 << 13)
	{
		mRegressionBlocks = 64;
	}
	else if (totalPoints <= 2 << 14)
	{
		mRegressionBlocks = 128;
	}
	else if (totalPoints <= 2 << 15)
	{
		mRegressionBlocks = 256;
	}
	else if (totalPoints <= 2 << 16)
	{
		mRegressionBlocks = 512;
	}
	else if (totalPoints <= 2 << 17)
	{
		mRegressionBlocks = 1024; // need y blocks past this point I believe. 
	}

	mRegressionTPB = totalPoints/mRegressionBlocks;
	std::cout << mSettings->windowWidth <<  " " << mSettings->windowHeight << std::endl;
	switch(mSettings->windowWidth * mSettings->windowHeight)
	{
		// 128 x 128
		case 16384:
			mColorTPB.x = mSettings->windowWidth/1;
			mColorTPB.y = mSettings->windowHeight/128;
			mColorBlocks.x = 1;
			mColorBlocks.y = 128;
			break;
		// 256 x 256
		case 65536:
			mColorTPB.x = mSettings->windowWidth/1;
			mColorTPB.y = mSettings->windowHeight/256;
			mColorBlocks.x = 1;
			mColorBlocks.y = 256;
			break;
		// 512 x 512
		case 262144:
			mColorTPB.x = mSettings->windowWidth/2;
			mColorTPB.y = mSettings->windowHeight/256;
			mColorBlocks.x = 2;
			mColorBlocks.y = 256;
			break;
		case 1024 * 1024:
			mColorTPB.x = mSettings->windowWidth/4;
			mColorTPB.y = mSettings->windowHeight/256;
			mColorBlocks.x = 4;
			mColorBlocks.y = 256;
			break;
		default:
			std::cout<<"Bad Dimensions"<<std::endl;
			exit(1);
	}
			
	std::cout<<"	Calling path algorithm kernels with:"<<std::endl
			 <<"	mRegressionTPB: ["<<mRegressionTPB<<"]"<<std::endl
			 <<"	On a Grid of: ["<<mRegressionBlocks<<"] Blocks"<<std::endl<<std::endl;
    std::cout<<"	Calling painting kernels with:"<<std::endl
    		 <<"	mColorTPB: ["<<mColorTPB.x<<","<<mColorTPB.y<<"]"<<std::endl
    		 <<"	On a Grid of: ["<<mColorBlocks.x<<","<<mColorBlocks.y<<"]"<<std::endl;
}

