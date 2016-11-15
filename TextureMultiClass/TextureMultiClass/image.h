//=========================================================================================================================//
// Created by
// Name : Aman Vora
// Email : amanvora@usc.edu
//=========================================================================================================================//

#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <stdio.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <direct.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern char inputPath[FILENAME_MAX];
extern char currentPath[FILENAME_MAX];
extern clock_t classificationTime, extractionTime;
extern clock_t funcStart;

unsigned char ***allocate3d(int row, int col, int bytesPerPixel);
float ***allocate3df(int row, int col, int bytesPerPixel);
unsigned char ***raw2array(char ipFilename[]);
unsigned char pixelValClamp(int pixelVal);
float ** calcFeatureVect(unsigned char ***imgArray);
void featureExtraction();
void performLDA();
void classifyTest();

#endif