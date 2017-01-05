#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "TanTriggs.h"

#include <iostream>

#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

// Normalize the image to the range of 0 to 255
Mat norm_0_255(const Mat& src) {
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

// Perform the Tan & Triggs preprocessing using the default parameters
Mat tan_triggs_preprocessing(Mat src) {

	float alpha = 0.1;
	float tau = 10.0;
	float gamma = 0.2; // For the gamma correction
	int sigma0 = 1; // DoG 
	int sigma1 = 2; // DoG 

	// Convert to floating point:
	Mat X = src;
	X.convertTo(X, CV_32FC1);

	/****************************************/
	// Start preprocessing:

	// Perform Gamma Correction:
	// Gamma correction essentially takes each pixel value and raise it to a certain power
	Mat I;
	I = X.clone();
	pow(X, gamma, I);
	// Perform Difference Of Gaussian:
	// DoG is essentially a subtraction of two Gaussian blurred images used to approximate the Laplacian of Gaussian
	// The DoG acts as a bandpass filter
	{
		Mat gaussian0, gaussian1;
		// Kernel Size:
		int kernel_sz0 = (3 * sigma0);
		int kernel_sz1 = (3 * sigma1);
		// Make them odd for OpenCV:
		kernel_sz0 += ((kernel_sz0 % 2) == 0) ? 1 : 0;
		kernel_sz1 += ((kernel_sz1 % 2) == 0) ? 1 : 0;
		GaussianBlur(I, gaussian0, Size(kernel_sz0, kernel_sz0), sigma0, sigma0, BORDER_REPLICATE);
		GaussianBlur(I, gaussian1, Size(kernel_sz1, kernel_sz1), sigma1, sigma1, BORDER_REPLICATE);
		subtract(gaussian0, gaussian1, I);
	}

	// Perform Contrast Equalization:
	// Create a standard measure of global contrast for all faces
	{
		double meanI = 0.0;
		{
			Mat tmp = I.clone();
			pow(abs(I), alpha, tmp);
			meanI = mean(tmp).val[0];

		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	{
		double meanI = 0.0;
		{
			Mat tmp = I.clone();
			pow(min(abs(I), tau), alpha, tmp);
			meanI = mean(tmp).val[0];
		}
		I = I / pow(meanI, 1.0 / alpha);
	}

	// Use the Hyperbolic Tangent function to to compress over-large values 
	// and reduce their influence on subsequent stages of processing
	{
		Mat exp_x, exp_negx;
		exp(I / tau, exp_x);
		exp(-I / tau, exp_negx);
		divide(exp_x - exp_negx, exp_x + exp_negx, I);
		I = tau * I;
	}
	
	// Bring the image back into the standard 0 to 255 range
	return norm_0_255(I);
}