#pragma once

#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2\face\facerec.hpp"
#include "opencv2\face.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

class PreProcess{
public:
	PreProcess(Mat image); // Preprocessing constructor

	Mat imageProcessed;
	Mat imagePrepare(Mat &image); // Gives access to the public image variable "imageProcessed
	Mat Output(); // Outputs the preprocessed image

private:
	Mat GrayConv(Mat original); // Converts the image to grayscale

	Mat Facial_HistEq(Mat face); // Applies a local histogram equalization

	Mat Face_ElipticalMask(Mat alignedFace, double width, double height); // Creates an elliptical mask around the face
};

