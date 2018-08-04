#include "SoftmaxInitializer.h"

SoftmaxInitializer::SoftmaxInitializer()
{

}

SoftmaxInitializer::~SoftmaxInitializer()
{
	std::cout << "Initializer Object Destroyed" << std::endl;
}

void SoftmaxInitializer::initialize(std::string aConfigFileName,
									std::shared_ptr<SoftmaxSettings>& aSoftmaxSettings,
 								    std::shared_ptr<SoftmaxData>& aSoftmaxData)
{
	_loadSettings(aConfigFileName, aSoftmaxSettings);
	// aSoftmaxData->hostPixelData.resize(3*aSoftmaxSettings->windowWidth*aSoftmaxSettings->windowHeight);
	// aSoftmaxData->devPixelData = aSoftmaxData->hostPixelData;
	_loadData(aSoftmaxSettings, aSoftmaxData);
}

	// I'm sure this is the most elegant way to read config file settings do not judge.

void SoftmaxInitializer::_loadSettings(std::string aConfigFileName,
				  			  		   std::shared_ptr<SoftmaxSettings>& aSettings)
{
	using namespace std;
	ifstream configFile;
	configFile.open(aConfigFileName);
	string line;
	std::cout<<"	Using Settings:"<<std::endl;
	while (getline(configFile, line))
	{
		if (line.find("numClasses") != std::string::npos) 
		{ 
			aSettings->numClasses = stoi(line.substr(10));
			std::cout<<"	numClasses: "<<aSettings->numClasses<<endl;
		}
		else if (line.find("numPoints") != std::string::npos) 
		{ 
			aSettings->numPoints = stoi(line.substr(9));
			std::cout<<"	numPoints: "<<aSettings->numPoints<<std::endl;
		}
		else if (line.find("stdDev") != std::string::npos)
		{ 
			aSettings->stdDev = stof(line.substr(6)); 
			std::cout<<"	stdDev: "<<aSettings->stdDev<<std::endl; 
		}
		else if (line.find("alphaX") != std::string::npos) 
		{ 
			aSettings->xAlphas = (float*)malloc(aSettings->numClasses * sizeof(float));
			std::cout<<"	xAlphas: " ;
			istringstream xAlphaStream(line);
			string dummy;
			xAlphaStream >> dummy;
			std::vector<std::string> strings {
				std::istream_iterator<std::string>(xAlphaStream), {}
			};
			std::vector<std::string>::iterator stringIter = strings.begin();

			int idx = 0;
			for (; stringIter != strings.end(); ++stringIter, ++idx)
			{
				aSettings->xAlphas[idx] = stof(*stringIter);
				std::cout << stof(*stringIter) << " ";
			}
			std::cout << std::endl;
		}
		else if (line.find("alphaY") != std::string::npos) 
		{ 
			aSettings->yAlphas = (float*)malloc(aSettings->numClasses * sizeof(float));
			std::cout<<"	yAlphas: " ;
			istringstream yAlphaStream(line);
			string dummy;
			yAlphaStream >> dummy;
			std::vector<std::string> strings {
				std::istream_iterator<std::string>(yAlphaStream), {}
			};
			//aSettings->xAlphas.resize(strings.size());
			int idx = 0;
			std::vector<std::string>::iterator stringIter = strings.begin();
			for (; stringIter != strings.end(); ++stringIter, ++idx)
			{
				aSettings->yAlphas[idx] = stof(*stringIter);
				std::cout << stof(*stringIter) << " ";
			}
			std::cout << std::endl;
		}
		else if (line.find("windowHeight") != std::string::npos)
		{
			aSettings->windowHeight = stoi(line.substr(12));
			std::cout<<"	windowHeight: "<<aSettings->windowHeight<<std::endl;
		}
		else if (line.find("windowWidth") != std::string::npos) 
		{ 
			aSettings->windowWidth = stoi(line.substr(11));
			std::cout<<"	windowWidth: "<<aSettings->windowWidth<<std::endl;
		}
		else if (line.find("numFeatures") != std::string::npos) 
		{ 
			aSettings->numFeatures = stoi(line.substr(11));
			std::cout<<"	numFeatures: "<<aSettings->numFeatures<<std::endl;
		}
		else if (line.find("plotNum") != std::string::npos) 
		{ 
			aSettings->plotNum = stoi(line.substr(7));
			std::cout<<"	plotNum: "<<aSettings->plotNum<<std::endl;
		}
		else if (line.find("recording") != std::string::npos)
		{
			aSettings->recording = bool(stoi(line.substr(9)));
			std::cout<<"	Recording: " << aSettings->recording << std::endl;
		}
		else if (line.find("frames") != std::string::npos)
		{
			aSettings->frames = stoi(line.substr(6));
			std::cout<<"	Frames: " << aSettings->frames << std::endl;
		}
	}
	configFile.close();
}

void SoftmaxInitializer::_loadData(std::shared_ptr<SoftmaxSettings>& aSettings,
		     				  	   std::shared_ptr<SoftmaxData>& aData)
{
	aData->hostPoints.resize(aSettings->numClasses * aSettings->numPoints);
	aData->hostQuadIndices.resize(aSettings->numClasses * aSettings->numPoints);
	aData->hostWeights.resize(aSettings->numClasses);
	aData->hostAlphas.resize(aSettings->numClasses);
	aData->hostDivLogLTerms.resize(aSettings->numClasses);
	aData->hostProbFields.resize(aSettings->numClasses);
	aData->hostPixelData.resize(3 * aSettings->windowWidth * aSettings->windowHeight);
	//std::cout << aData->hostPixelData.size() << " " << &aData->hostPixelData << " " << &aData->devPixelData << " " << aData->hostPixelData.size() << std::endl;
	//thrust::fill(aData->hostPixelData.begin(), aData->hostPixelData.end(), (uint8_t)0);

	std::cout << "happes" <<std::endl;
	//aData->devPixelData=aData->hostPixelData;

	// populate point data with gaussian distributions based on settings
	_genGaussians(aSettings, aData->hostPoints);
	thrust::fill(aData->hostQuadIndices.begin(), aData->hostQuadIndices.end(), 0);
	thrust::fill(aData->hostWeights.begin(), aData->hostWeights.end(), make_float2(0.0, 0.0));

	aData->devDivLogLTerms.resize(aSettings->numClasses);
	aData->devDivLogLPtrs.resize(aSettings->numClasses);
	aData->devProbFields.resize(aSettings->numClasses);
	aData->devProbFieldPtrs.resize(aSettings->numClasses);
	aData->devPixelData.resize(3 * aSettings->windowWidth * aSettings->windowHeight);
	std::cout << "happened" << std::endl;
	for (int classIdx = 0; classIdx < aSettings->numClasses; classIdx++)
	{
		aData->hostDivLogLTerms[classIdx].resize(aSettings->numFeatures);
		aData->devDivLogLTerms[classIdx].resize(aSettings->numFeatures);
		aData->devDivLogLPtrs[classIdx].resize(aSettings->numFeatures);
		aData->hostProbFields[classIdx].resize(aSettings->windowWidth * aSettings->windowHeight);
		for (int f = 0; f < aSettings->numFeatures; f++)
		{
			aData->hostDivLogLTerms[classIdx][f].resize(aSettings->numPoints * aSettings->numClasses);
			thrust::fill(aData->hostDivLogLTerms[classIdx][f].begin(), aData->hostDivLogLTerms[classIdx][f].end(), 0.0);
			aData->devDivLogLTerms[classIdx][f] = aData->hostDivLogLTerms[classIdx][f];
			aData->devDivLogLPtrs[classIdx][f] = thrust::raw_pointer_cast(aData->devDivLogLTerms[classIdx][f].data());
		}
	}

		// set each x, y, and z alpha to the one found in the settings.
		// nothing variable about them just yet.

	for (int classIdx = 0;classIdx < aSettings->numClasses; classIdx++)
	{
		aData->hostAlphas[classIdx] = make_float2(aSettings->xAlphas[classIdx],aSettings->yAlphas[classIdx]);
	}
	std::cout << "happened" << std::endl;

		// let thrust do cudaMemCopy for us
	aData->devPoints = aData->hostPoints;
	aData->devQuadIndices = aData->hostQuadIndices;
	aData->devWeights = aData->hostWeights;
	std::cout << "happened" << std::endl;
	//aData->devPixelData = aData->hostPixelData;
	std::cout << "happened????" << std::endl;

		// and get raw data pointers for the device kernels
	aData->devPointsPtr = thrust::raw_pointer_cast(aData->devPoints.data());
	aData->devQuadIndicesPtr = thrust::raw_pointer_cast(aData->devQuadIndices.data());
	aData->devWeightsPtr = thrust::raw_pointer_cast(aData->devWeights.data());
	std::cout << "happened" << std::endl;	
	aData->devPixelDataPtr = thrust::raw_pointer_cast(aData->devPixelData.data());

	for (int classIdx = 0;classIdx < aSettings->numClasses; classIdx++)
	{
		aData->devProbFields[classIdx].resize(aSettings->windowWidth * aSettings->windowHeight);
		aData->devProbFields[classIdx] = aData->hostProbFields[classIdx];
		aData->devProbFieldPtrs[classIdx] = thrust::raw_pointer_cast(aData->devProbFields[classIdx].data());
	}
	std::cout << "happened" << std::endl;

	aData->hostColorMap.resize(512);
	std::ifstream colorfile("data/Hot_Cold_No_Zero", std::ifstream::in);
	std::string colorLine;
	int i = 0;
	while(getline(colorfile, colorLine)){
		std::stringstream linestream(colorLine);
		linestream >> aData->hostColorMap[i].x >> aData->hostColorMap[i].y >> aData->hostColorMap[i].z;
		i++;
	}
	colorfile.close();

	aData->devColorMap = aData->hostColorMap;
	aData->devColorMapPtr = thrust::raw_pointer_cast(aData->devColorMap.data());
}

void SoftmaxInitializer::_genGaussians(std::shared_ptr<SoftmaxSettings>& aSettings,
	     				  		  	   thrust::host_vector<float2>& hostPoints)
{
		// Some day I will cli this
		// could default to equispaced offsets on a sphere
		// unless something more complicated is desired.

	std::vector<float2> gaussOffsets; 
	if (aSettings->numClasses == 8)
	{
		gaussOffsets.push_back(make_float2( 0.0, 0.5));
		gaussOffsets.push_back(make_float2( 0.5, 0.5));
		gaussOffsets.push_back(make_float2( 0.5, 0.0));
		gaussOffsets.push_back(make_float2( 0.5,-0.5));
		gaussOffsets.push_back(make_float2( 0.0,-0.5));
		gaussOffsets.push_back(make_float2(-0.5,-0.5));
		gaussOffsets.push_back(make_float2(-0.5, 0.0));
		gaussOffsets.push_back(make_float2(-0.5, 0.5));
	}
	else if (aSettings->numClasses == 4)
	{
		gaussOffsets.push_back(make_float2( 0.85, 0.85));
		gaussOffsets.push_back(make_float2( 0.65,-0.5));
		gaussOffsets.push_back(make_float2(-0.45,-0.35));
		gaussOffsets.push_back(make_float2(-0.5, 0.35));

		// gaussOffsets.push_back(make_float2( 0.85, 0.85));
		// gaussOffsets.push_back(make_float2( 0.35, 0.35));
		// gaussOffsets.push_back(make_float2(-0.45,-0.35));
		// gaussOffsets.push_back(make_float2(-0.5, 0.35));

		// gaussOffsets.push_back(make_float2( 0.5, 0.0));
		// gaussOffsets.push_back(make_float2( 0.0,-0.5));
		// gaussOffsets.push_back(make_float2(-0.5,-0.0));
		// gaussOffsets.push_back(make_float2(-0.0, 0.5));
	}
	else if (aSettings->numClasses == 2)
	{
		gaussOffsets.push_back(make_float2(-0.5,-0.5));
		gaussOffsets.push_back(make_float2( 0.5, 0.5));	
	}

	float gaussMean = 0.0;					// Setting, TODO Config->CLI handler
	
	float gaussStdDev = aSettings->stdDev; 	// Setting, TODO Config->CLI handler

	std::default_random_engine gen;
  	std::normal_distribution<float> normDist(gaussMean,gaussStdDev);

  		// have to promise same size
  	
  	int pointIdx = 0;
	for (int classIdx = 0; classIdx < gaussOffsets.size(); classIdx++)
	{
		for(; pointIdx < aSettings->numPoints;  pointIdx++)
		{
			hostPoints[pointIdx + classIdx * aSettings->numPoints].x = normDist(gen) + gaussOffsets[classIdx].x;
			hostPoints[pointIdx + classIdx * aSettings->numPoints].y = normDist(gen) + gaussOffsets[classIdx].y;
		}
		pointIdx = 0;
	}
}
