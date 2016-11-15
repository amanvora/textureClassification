//=========================================================================================================================//
// Created by
// Name : Aman Vora
// Email : amanvora@usc.edu
//=========================================================================================================================//
#include "image.h"

using namespace cv;
using namespace std;

int main()
{
	cout << "Texture Classification of multiple classes using LDA" << endl;

	_getcwd(currentPath, sizeof(currentPath));
	strcpy(inputPath, currentPath);
	strcat(inputPath, "/Images/P1/part b");

	featureExtraction();
	performLDA();
	classifyTest();

	cout << "Average time for feature extraction per image: " << ((float)extractionTime) / CLOCKS_PER_SEC << " s." << endl;
	cout << "Average time for recognition per test image: " << ((float)classificationTime) / CLOCKS_PER_SEC << " ms." << endl;
	system("Pause");
	return 0;
}