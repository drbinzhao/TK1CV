#include "SPImage.hpp"
#include "omp.h"
//#include <mpi.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>

#define NUMBER 1

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  cuda::DeviceInfo gpu;
  if (!gpu.isCompatible()) {
    cout << "GPU is not compatible\n";
    exit(-1);
  }
  SPImage generator;
  Mat image[25];
  for (int i = 0; i < NUMBER; i++) {
    image[i] = generator.Generate(4096, 1024);
    string fileName = "imgO" + to_string(i) + ".bmp";
    imwrite(fileName, image[i]);
  }
  cuda::GpuMat gImage[25];
  for (int i = 0; i < NUMBER; i++) {
    gImage[i].upload(image[i]);
  }
  Ptr<cuda::Filter> gpuBlur = cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(5, 5));
  Mat imageOut[25];
  cuda::GpuMat gOut[25];
  Mat gImageOut[25];
  Mat mpImageOut[25];

  double sStart = (double) getTickCount();
  for (int i = 0; i < NUMBER; i++) {
    blur(image[i], imageOut[i], Size(5, 5));
  }
  double sTime = ((double) getTickCount() - sStart) / getTickFrequency();
  cout << sTime << "\n";

  double mpStart = (double) getTickCount();
#pragma omp parallel for
  for (int i = 0; i < NUMBER; i++) {
    blur(image[i], mpImageOut[i], Size(5, 5));
  }
  double mpTime = ((double) getTickCount() - mpStart) / getTickFrequency();
  cout << mpTime << "\n";

  double gStart = (double) getTickCount();
  for (int i = 0; i < NUMBER; i++) {
    gpuBlur->apply(gImage[i], gOut[i]);
  }
  double gTime = ((double) getTickCount() - gStart) / getTickFrequency();
  cout << gTime << "\n";

  for (int i = 0; i < 25; i++) {
    gOut[i].download(gImageOut[i]);
  }

  for (int i = 0; i < NUMBER; i++) {
    string sName = "imgS" + to_string(i) + ".bmp";
    string gName = "imgG" + to_string(i) + ".bmp";
    string mpName = "imgMP" + to_string(i) + ".bmp";
    imwrite(sName, imageOut[i]);
    imwrite(gName, gImageOut[i]);
    imwrite(mpName, mpImageOut[i]);
  }
}
