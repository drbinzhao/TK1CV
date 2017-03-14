/*****
 * Created by DS
 * Implementation of SPImage
 *****/
#include "SPImage.hpp"
#include <opencv2/opencv.hpp>

using namespace cv;

Mat SPImage::Generate(int width, int height) {
  // Generate a image with width, height. Initial value is 128.
  Mat newImage(Size(width, height), CV_8UC1, Scalar(128));
  // Add Salt and pepper noise
  setRNGSeed((int) getTickCount());
  randu(newImage, 0, 255);
  return newImage;
}

Mat* SPImage::Generate100(int width, int height) {
  for (int i = 0; i < 100; i++) {
    _hundredMat[i] = this->Generate(width, height);
  }
  return _hundredMat;
}
