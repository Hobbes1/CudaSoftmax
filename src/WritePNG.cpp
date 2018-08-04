#include "WritePNG.h"
int WritePNG(unsigned char* data, std::string fileName, unsigned int width, unsigned int height)
{

	FILE *pngFile = fopen(fileName.c_str(), "wb");
  std::cout << "creating png image: " << fileName << std::endl;
  png_structp png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, 
  											  nullptr,
  											  nullptr,
  											  nullptr);
  if (!png_ptr)
	fprintf(stderr, "ERROR: png_ptr not created\n");

  png_infop info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
  {
     png_destroy_write_struct(&png_ptr,
       (png_infopp)NULL);
     fprintf(stderr, "ERROR: into_ptr not created\n");
  }

  png_init_io(png_ptr, pngFile);
 	png_set_IHDR(png_ptr, info_ptr, width, height,
  8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
  PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

  png_write_info(png_ptr, info_ptr);

  png_bytep* row_pointers = (png_bytep*)malloc(sizeof(png_bytep)*height);
  for (int i = 0; i <  height; i++)
  {
    row_pointers[i] = data + i * 3 * width;
    //std::cout << "Row Pointer starts at val: " << i*width << " "<< (int)data[i*width] << std::endl;
  }

  png_set_rows(png_ptr, info_ptr, row_pointers);
  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, row_pointers);

	if (pngFile != NULL) fclose(pngFile);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
  // WARNING Calling code should free its own data pointer when ready to do so. 
  //if (data != NULL) free(data);		
}


std::string FrameNameGen(int frameIdx, int totalFrames)
{
	int numZeros = std::to_string(totalFrames).length() + 1 - std::to_string(frameIdx).length();
	std::string stepsString = std::string(numZeros, '0') + std::to_string(frameIdx);
	std::string fileName = "video/frame_" + stepsString + ".png";
	return fileName;
}