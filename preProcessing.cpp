#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"

#include <iostream> // For standard input and output
#include <fstream> // For files
#include <sstream> // For strings
#include "preProcessing.h"
#include "opencv2\face\facerec.hpp"
#include "opencv2\face.hpp"

using namespace cv;
using namespace cv::face;
using namespace std;

//---------------------------------------------------------------------------------------------------------------------------------
PreProcess::PreProcess(Mat image){
	imageProcessed = imagePrepare(image);
	imageProcessed = GrayConv(imageProcessed);
	imageProcessed = Facial_HistEq(imageProcessed);
	//imageProcessed = Face_ElipticalMask(imageProcessed, imageProcessed.cols, imageProcessed.rows);
}

//----------------------------------------------------------------------------------------------------------------------------------

// Allows us to pass the image between functions
// --------------------------------------------
Mat PreProcess::imagePrepare(Mat &image)
{
	imageProcessed = image;
	return imageProcessed;
}

//--------------------------------------------------------------------------------------------------------------------------------

Mat PreProcess::GrayConv(Mat original){
	Mat gray;
	if (original.channels() == 3) {
		cvtColor(original, gray, CV_BGR2GRAY);
	}
	else if (
		original.channels() == 4) {
		cvtColor(original, gray, CV_BGRA2GRAY);
	}
	else {
		// Access the grayscale input image directly.
		gray = original;
	}
	return gray;
}
//----------------------------------------------------------------------------------------------------------------------------------

Mat PreProcess::Facial_HistEq(Mat face){

	// It is common that there is stronger light from one half of the face than the other. In that case,
	// if you simply did histogram equalization on the whole face then it would make one half dark and
	// one half bright. So we will do histogram equalization separately on each face half, so they will
	// both look similar on average. But this would cause a sharp edge in the middle of the face, because
	// the left half and right half would be suddenly different. So we also histogram equalize the whole
	// image, and in the middle part we blend the 3 images together for a smooth brightness transition.

	// First we apply histogram equalization to the whole face
	int w = face.cols;
	int h = face.rows;
	Mat wholeFace;
	equalizeHist(face, wholeFace);
	// Now we sperate the left and right halves of the face and apply histogram equalization to them individually
	int midX = w / 2;
	Mat leftSide = face(Rect(0, 0, midX, h));
	Mat rightSide = face(Rect(midX, 0, w - midX, h));
	equalizeHist(leftSide, leftSide);
	equalizeHist(rightSide, rightSide);

	// Now we combine the three images together by accessing the pixels of each image directly
	for (int y = 0; y<h; y++) {
		for (int x = 0; x<w; x++) {
			int v;
			if (x < w / 4) {
				// Left 25%: just use the left face.
				v = leftSide.at<uchar>(y, x);
			}
			else if (x < w * 2 / 4) {
				// Mid-left 25%: blend the left face & whole face.
				int lv = leftSide.at<uchar>(y, x);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the whole face as it moves
				// further right along the face.
				float f = (x - w * 1 / 4) / (float)(w / 4);
				v = cvRound((1.0f - f) * lv + (f)* wv);
			}
			else if (x < w * 3 / 4) {
				// Mid-right 25%: blend right face & whole face.
				int rv = rightSide.at<uchar>(y, x - midX);
				int wv = wholeFace.at<uchar>(y, x);
				// Blend more of the right-side face as it moves
				// further right along the face.
				float f = (x - w * 2 / 4) / (float)(w / 4);
				v = cvRound((1.0f - f) * wv + (f)* rv);
			}
			else {
				// Right 25%: just use the right face.
				v = rightSide.at<uchar>(y, x - midX);
			}
			face.at<uchar>(y, x) = v;
		}// end x loop
	}//end y loop

	return face;
}

//---------------------------------------------------------------------------------------------------------------------------------------

Mat PreProcess::Face_ElipticalMask(Mat alignedFace, double width, double height){

	// Draw a black-filled ellipse in the middle of the image.
	// First we initialize the mask image to white (255).
	Mat mask = Mat(alignedFace.size(), CV_8UC1, Scalar(255));

	Point faceCenter = Point(cvRound(width * 0.5), cvRound(height * 0.4));
	Size size = Size(cvRound(width * 0.5), cvRound(height * 0.8));
	ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(0), CV_FILLED);
	// Apply the elliptical mask on the face, to remove corners.
	// Sets corners to gray, without touching the inner face.
	alignedFace.setTo(Scalar(128), mask);

	return alignedFace;
}

//---------------------------------------------------------------------------------------------------
Mat PreProcess::Output(){

	return imageProcessed;
}
