/*************
 * A opencv test for creating 100 image with salt & pepper noise, then use boxfilter, all in parallel.
 * This test is for comparing compute power of ODroid and T5810.
 * Created by DS
 *************/
#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class SPImage {
public:
  Mat Generate(int width, int height);
  Mat* Generate100(int width, int height);
private:
  Mat _hundredMat[100];
};
