//=========================================================================================================================//
// Created by
// Name : Aman Vora
// Email : amanvora@usc.edu
//=========================================================================================================================//

#include "image.h"

using namespace cv;
using namespace std;

// Define file pointer and variables
FILE *file;
float featureAvgP1;
float featureAvgP2;
Mat projectedTrainFV, projectedTestFV;

int posClass[48];
int negClass[48];
char inputPath[FILENAME_MAX];
char currentPath[FILENAME_MAX];
clock_t classificationTime, extractionTime;
clock_t funcStart;

int row = 128, col = 128, bytesPerPixel = 1;
float ***lawsKernels = allocate3df(5, 5, 25);
float ***meanArray = allocate3df(5, 5, 1);
float ***lawsFilterArr = allocate3df(5, 5, 25);
float ***featureVectP1 = allocate3df(36, 25, 1); //Store the feature vectors of the images with pattern 1
float ***featureVectP2 = allocate3df(36, 25, 1); //Store the feature vectors of the images with pattern 2
float ***featureVectTest = allocate3df(24, 25, 1); //Store the feature vectors of the images with unlabelled pattern
int recognized_pattern[24] = {};

//=========================================================================================================================//
unsigned char ***allocate3d(int row, int col, int bytesPerPixel)
{
	unsigned char ***array3d = new unsigned char **[row * sizeof(unsigned char *)];
	for (int i = 0; i < row; i++)
	{
		array3d[i] = new unsigned char *[col * sizeof(unsigned char *)];
		for (int j = 0; j < col; j++)
			array3d[i][j] = new unsigned char[bytesPerPixel * sizeof(unsigned char)];
	}
	return array3d;
}
//=========================================================================================================================//
float ***allocate3df(int row, int col, int bytesPerPixel)
{
	float ***array3d = new float **[row * sizeof(float *)];
	for (int i = 0; i < row; i++)
	{
		array3d[i] = new float *[col * sizeof(float *)];
		for (int j = 0; j < col; j++)
			array3d[i][j] = new float[bytesPerPixel * sizeof(float)];
	}
	return array3d;
}
//=========================================================================================================================//
unsigned char ***raw2array(char ipFilename[])
{
	unsigned char *raw = new unsigned char[row*col*bytesPerPixel * sizeof(unsigned char *)];
	unsigned char ***imgArray = allocate3d(row, col, bytesPerPixel);

	_chdir(inputPath);

	// Read image (filename specified by first argument) into image data matrix
	if (!(file = fopen(ipFilename, "rb")))
	{
		cout << "Cannot open file: " << ipFilename << endl;
		exit(1);
	}
	fread(raw, sizeof(unsigned char), row*col*bytesPerPixel, file);
	fclose(file);

	long int rawCount = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int k = 0; k < bytesPerPixel; k++)
				imgArray[i][j][k] = (raw[rawCount++]);
		}
	}
	return imgArray;
}
//=========================================================================================================================//
unsigned char pixelValClamp(int pixelVal)
{
	if (pixelVal < 0)
		pixelVal = 0;
	else
		if (pixelVal > 255)
			pixelVal = 255;

	return pixelVal;
}
//=========================================================================================================================//
float ** calcFeatureVect(unsigned char ***imgArray)
{
	float ***featureVectImg = allocate3df(1, 25, 1);
	float ***zeroMeanArr = allocate3df(row + 4, col + 4, bytesPerPixel);
	unsigned char ***bufferArr = allocate3d(row + 4, col + 4, bytesPerPixel);

	//------------------------------------------------------------------------------------------//
	// Boundary Extension
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			for (int k = 0; k < bytesPerPixel; k++)
			{
				bufferArr[i + 2][j + 2][k] = imgArray[i][j][k];
			}
		}
	}

	// Extend the rows
	for (int j = 0; j < col + 4; j++) {
		for (int k = 0; k < bytesPerPixel; k++) {
			bufferArr[0][j][k] = bufferArr[2][j][k];
			bufferArr[1][j][k] = bufferArr[2][j][k];
			bufferArr[row + 3][j][k] = bufferArr[row + 1][j][k];
			bufferArr[row + 2][j][k] = bufferArr[row + 1][j][k];
		}
	}

	// Extend the columns
	for (int i = 0; i < row + 4; i++) {
		for (int k = 0; k < bytesPerPixel; k++) {
			bufferArr[i][0][k] = bufferArr[i][2][k];
			bufferArr[i][1][k] = bufferArr[i][2][k];
			bufferArr[i][col + 3][k] = bufferArr[i][col + 1][k];
			bufferArr[i][col + 2][k] = bufferArr[i][col + 1][k];
		}
	}

	//------------------------------------------------------------------------------------------//
	// Calculate zero mean array

	for (int i = 2; i < row + 2; i++) {
		for (int j = 2; j < col + 2; j++) {
			//Calculate local zero mean array
			//Calculate local value of mean
			float localZeroMeanArr[5][5];
			float mu;
			float meanSum = 0;
			for (int local_i = 0; local_i < 5; local_i++) {
				for (int local_j = 0; local_j < 5; local_j++) {
					meanSum += bufferArr[(i - 2) + local_i][(j - 2) + local_j][0];
				}
			}
			mu = meanSum / 25;

			//Subtract local mean from the window
			zeroMeanArr[i][j][0] = float(bufferArr[i][j][0]) - mu;
		}
	}

	// Extend the rows
	for (int j = 0; j < col + 4; j++) {
		for (int k = 0; k < bytesPerPixel; k++) {
			zeroMeanArr[0][j][k] = zeroMeanArr[2][j][k];
			zeroMeanArr[1][j][k] = zeroMeanArr[2][j][k];
			zeroMeanArr[row + 3][j][k] = zeroMeanArr[row + 1][j][k];
			zeroMeanArr[row + 2][j][k] = zeroMeanArr[row + 1][j][k];
		}
	}

	// Extend the columns
	for (int i = 0; i < row + 4; i++) {
		for (int k = 0; k < bytesPerPixel; k++) {
			zeroMeanArr[i][0][k] = zeroMeanArr[i][2][k];
			zeroMeanArr[i][1][k] = zeroMeanArr[i][2][k];
			zeroMeanArr[i][col + 3][k] = zeroMeanArr[i][col + 1][k];
			zeroMeanArr[i][col + 2][k] = zeroMeanArr[i][col + 1][k];
		}
	}

	//------------------------------------------------------------------------------------------//
	float sum_pixel_fv[25] = {};
	for (int i = 4; i < row; i++) {
		for (int j = 4; j < col; j++) {
			for (int local_k = 0; local_k < 25; local_k++) {
				float energyPixel = 0;
				for (int local_i = 0; local_i < 5; local_i++) {
					for (int local_j = 0; local_j < 5; local_j++) {
						lawsFilterArr[local_i][local_j][local_k] = zeroMeanArr[(i - 2) + local_i][(j - 2) + local_j][0] *
							lawsKernels[local_i][local_j][local_k];
						energyPixel += pow((lawsFilterArr[local_i][local_j][local_k]), 2) / 25;
					}
				}
				sum_pixel_fv[local_k] += energyPixel;
			}
		}
	}

	for (int j = 0; j < 25; j++)
		featureVectImg[0][j][0] = sum_pixel_fv[j] / (row*col);

	for (int i = 0; i < row; i++){
		for (int j = 0; j < col; j++) {
			delete[](bufferArr[i][j]);
		}
		delete[](bufferArr[i]);
	}
	delete[](bufferArr);
	return featureVectImg[0];
}
//=========================================================================================================================//
void featureExtraction()
{
	funcStart = clock();
	char ipFilename[20];
	char imgNumber[3];
	unsigned char ***ipArr = allocate3d(row, col, bytesPerPixel);
	//------------------------------------------------------------------------------------------//
	// Define the 5 1-D kerenls
	int laws_1d[5][5] =
	{
		{ 1, 4, 6, 4, 1 },	// L5 (Level)
		{ -1, -2, 0, 2, 1 },	// E5 (Edge)
		{ -1, 0, 2, 0, -1 },	// S5 (Spot)
		{ -1, 2, 0, -2, 1 },	// W5 (Wave)
		{ 1, -4, 6, -4, 1 }	// R5 (Ripple)
	};
	//------------------------------------------------------------------------------------------//
	// Calculate the 25 Laws' 2D kernels
	for (int k = 0; k < 25; k++) {
		for (int i = 0; i < 5; i++)	{
			for (int j = 0; j < 5; j++)	{
				lawsKernels[i][j][k] = laws_1d[k / 5][i] * laws_1d[k % 5][j];
			}
		}
	}
	//------------------------------------------------------------------------------------------//
	// Generate a random sequence for preferred pattern
	srand(time(NULL));
	int generateClassIndex[48];

	for (int i = 1; i <= 48; i++)
		generateClassIndex[i - 1] = i;

	std::random_shuffle(std::begin(generateClassIndex), std::end(generateClassIndex));
	// This indicates all the training iamge files in the positive class
	for (int i = 0; i < 36; i++)
		posClass[i] = generateClassIndex[i];

	// This indicates 12 of the testing image files
	for (int i = 36; i < 48; i++)
		posClass[i] = generateClassIndex[i];

	for (int j = 0; j < 3; j++)
	{
		std::random_shuffle(std::begin(generateClassIndex), std::end(generateClassIndex));
		for (int i = 0; i < 16; i++)
			negClass[i + 16*j] = generateClassIndex[i];
	}

	//------------------------------------------------------------------------------------------//
	// Find the feature vectors of images from the positive class
	//------------------------------------------------------------------------------------------//
	cout << "Training images of positive class" << endl;
	for (int i = 0; i < 36; i++)
	{
		imgNumber[0] = char((posClass[i] / 10) + 48);
		imgNumber[1] = char((posClass[i] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "grass_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectP1[i] = calcFeatureVect(ipArr);

		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;

	//------------------------------------------------------------------------------------------//
	// Find the feature vectors of images from the negative class
	//------------------------------------------------------------------------------------------//
	cout << "Training images of negative class" << endl;
	for (int i = 0; i < 12; i++)
	{
		imgNumber[0] = char((negClass[i] / 10) + 48);
		imgNumber[1] = char((negClass[i] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "straw_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectP2[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	for (int i = 12; i < 24; i++)
	{
		imgNumber[0] = char((negClass[i] / 10) + 48);
		imgNumber[1] = char((negClass[i] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "leather_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectP2[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	for (int i = 24; i < 36; i++)
	{
		imgNumber[0] = char((negClass[i] / 10) + 48);
		imgNumber[1] = char((negClass[i] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "sand_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectP2[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	//------------------------------------------------------------------------------------------//
	cout << "Testing images" << endl;
	// Find the feature vectors of test images
	for (int i = 0; i < 12; i++)
	{
		imgNumber[0] = char((posClass[i + 36] / 10) + 48);
		imgNumber[1] = char((posClass[i + 36] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "grass_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectTest[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	for (int i = 0; i < 4; i++)
	{
		imgNumber[0] = char((negClass[i + 12] / 10) + 48);
		imgNumber[1] = char((negClass[i + 12] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "straw_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectTest[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	for (int i = 0; i < 4; i++)
	{
		imgNumber[0] = char((negClass[i + 28] / 10) + 48);
		imgNumber[1] = char((negClass[i + 28] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "leather_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectTest[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	for (int i = 0; i < 4; i++)
	{
		imgNumber[0] = char((negClass[i + 44] / 10) + 48);
		imgNumber[1] = char((negClass[i + 44] % 10) + 48);
		imgNumber[2] = '\0';
		strcpy(ipFilename, "sand_");
		strcat(ipFilename, imgNumber);
		strcat(ipFilename, ".raw");
		ipArr = raw2array(ipFilename);
		cout << (i+1) << ". " << "Obtaining features of " << ipFilename << endl;
		featureVectTest[i] = calcFeatureVect(ipArr);
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++) {
				delete[](ipArr[i][j]);
			}
			delete[](ipArr[i]);
		}
		delete[](ipArr);
	}
	cout << endl;
	//------------------------------------------------------------------------------------------//
	extractionTime = (clock() - funcStart) / 96;
}
//=========================================================================================================================//
void performLDA()
{
	Mat trainFV(72, 25, CV_32F);
	Mat labelVector(72, 1, CV_32F);
	Mat testFV(24, 25, CV_32F);
	//------------------------------------------------------------------------------------------//
	for (int i = 0; i < 36; i++)
	{
		for (int j = 0; j < 25; j++)
		{
			trainFV.at<float>(i, j) = featureVectP1[i][j][0];
			trainFV.at<float>(i + 36, j) = featureVectP2[i][j][0];
		}
		labelVector.at<float>(i, 0) = 0;
		labelVector.at<float>(i + 36, 0) = 1;
	}

	for (int i = 0; i < 24; i++)
	{
		for (int j = 0; j < 25; j++)
		{
			testFV.at<float>(i, j) = featureVectTest[i][j][0];
		}
	}

	LDA lda(1);
	lda.compute(trainFV, labelVector);
	projectedTrainFV = lda.project(trainFV);
	projectedTestFV = lda.project(testFV);

	//------------------------------------------------------------------------------------------//
	// Obtain mean of all the feature vectors in training pattern
	float sumFV_P1 = 0;
	float sumFV_P2 = 0;
	for (int j = 0; j < 36; j++)
	{
		sumFV_P1 += projectedTrainFV.at<double>(j);
		sumFV_P2 += projectedTrainFV.at<double>(j + 36);
	}
	featureAvgP1 = sumFV_P1 / 36;
	featureAvgP2 = sumFV_P2 / 36;
	//------------------------------------------------------------------------------------------//
}
//=========================================================================================================================//
void classifyTest()
{
	funcStart = clock();
	char ipFilename[20];
	char imgNumber[3];
	float distP1 = 0, distP2 = 0;
	int successCount = 0;

	int fVNumRow = 24;
	int fVNumCol = 1;
	Mat fvPat1(fVNumRow, fVNumCol, CV_32F, Scalar(0));
	Mat fvPat2(fVNumRow, fVNumCol, CV_32F, Scalar(0));
	Mat points(2, fVNumCol, CV_32F, Scalar(0));
	Mat mean, covarGrass, covarStraw, invcovarGrass, invcovarStraw;

	for (int j = 0; j < fVNumRow; j++)
	{
		for (int k = 0; k < fVNumCol; k++)
		{
			fvPat1.at<float>(j, k) = projectedTrainFV.at<double>(j, k);
			fvPat2.at<float>(j, k) = projectedTrainFV.at<double>(j + 36, k);
		}
	}
	//------------------------------------------------------------------------------------------//
	calcCovarMatrix(fvPat1, covarGrass, mean, CV_COVAR_NORMAL + CV_COVAR_ROWS, -1);
	covarGrass = covarGrass / (fvPat1.rows - 1);
	invert(covarGrass, invcovarGrass, DECOMP_SVD);

	calcCovarMatrix(fvPat2, covarStraw, mean, CV_COVAR_NORMAL + CV_COVAR_ROWS, -1);
	covarStraw = covarStraw / (fvPat2.rows - 1);
	invert(covarStraw, invcovarStraw, DECOMP_SVD);

	//------------------------------------------------------------------------------------------//
	// Classification of the testing images
	//------------------------------------------------------------------------------------------//
	successCount = 0;
	for (int i = 0; i < fVNumRow; i++)
	{
		for (int j = 0; j < fVNumCol; j++)
		{
			points.at<float>(0, j) = featureAvgP1;
			points.at<float>(1, j) = projectedTestFV.at<double>(i, j);
		}
		distP1 = Mahalanobis(points(Range(0, 1), Range::all()), points(Range(1, 2), Range::all()), invcovarGrass);
		//------------------------------------------------------------------------------------------//
		for (int j = 0; j < fVNumCol; j++)
		{
			points.at<float>(0, j) = featureAvgP2;
			points.at<float>(1, j) = projectedTestFV.at<double>(i, j);
		}
		distP2 = Mahalanobis(points(Range(0, 1), Range::all()), points(Range(1, 2), Range::all()), invcovarStraw);
		//------------------------------------------------------------------------------------------//
		if (i < 12) {
			imgNumber[0] = char((posClass[i + 36] / 10) + 48);
			imgNumber[1] = char((posClass[i + 36] % 10) + 48);
			imgNumber[2] = '\0';
			strcpy(ipFilename, "grass_");
			strcat(ipFilename, imgNumber);
			strcat(ipFilename, ".raw");
			cout << (i+1) << ". " << ipFilename << " recognized as ";
			if (distP1 < distP2)
			{
				cout << "grass" << endl;
				successCount++;
			}
			else
			{
				cout << "non-grass" << endl;
			}
		}
		//------------------------------------------------------------------------------------------//
		if (i >= 12 && i < 16) {
			imgNumber[0] = char((negClass[(i-12) + 12] / 10) + 48); 
			// Left as such so that the test image in question is easily found
			imgNumber[1] = char((negClass[(i-12) + 12] % 10) + 48);
			imgNumber[2] = '\0';
			strcpy(ipFilename, "straw_");
			strcat(ipFilename, imgNumber);
			strcat(ipFilename, ".raw");
			cout << (i+1) << ". " << ipFilename << " recognized as ";
			if (distP1 < distP2)
			{
				cout << "grass" << endl;
			}
			else
			{
				cout << "non-grass" << endl;
				successCount++;
			}
		}
		//------------------------------------------------------------------------------------------//
		if (i >= 16 && i < 20) {
			imgNumber[0] = char((negClass[(i-16) + 28] / 10) + 48); 
			imgNumber[1] = char((negClass[(i-16) + 28] % 10) + 48);
			imgNumber[2] = '\0';
			strcpy(ipFilename, "leather_");
			strcat(ipFilename, imgNumber);
			strcat(ipFilename, ".raw");
			cout << (i+1) << ". " << ipFilename << " recognized as ";
			if (distP1 < distP2)
			{
				cout << "grass" << endl;
			}
			else
			{
				cout << "non-grass" << endl;
				successCount++;
			}
		}
		//------------------------------------------------------------------------------------------//
		if (i >= 20 && i < 24) {
			imgNumber[0] = char((negClass[(i-20) + 44] / 10) + 48);
			imgNumber[1] = char((negClass[(i-20) + 44] % 10) + 48);
			imgNumber[2] = '\0';
			strcpy(ipFilename, "sand_");
			strcat(ipFilename, imgNumber);
			strcat(ipFilename, ".raw");
			cout << (i+1) << ". " << ipFilename << " recognized as ";
			if (distP1 < distP2)
			{
				cout << "grass" << endl;
			}
			else
			{
				cout << "non-grass" << endl;
				successCount++;
			}
		}
	}
	cout << "Recognition rate of the training data is : "
		 << (successCount / float(fVNumRow)) * 100 << "%" << endl << endl;
	//------------------------------------------------------------------------------------------//
	classificationTime = (clock() - funcStart) * 1000 / float(fVNumRow);
	//------------------------------------------------------------------------------------------//
	return;
}
//=========================================================================================================================//