#pragma once

#include <opencv2/core.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2\face\facerec.hpp"
#include "opencv2\face.hpp"

using namespace std;

class VideoFaceDetector
{
public:
	VideoFaceDetector(const std::string cascadeFilePath, cv::VideoCapture &videoCapture); // Video capturing constructor
	~VideoFaceDetector(); // Video capturing destructor

	cv::Point               getFrameAndDetect(cv::Mat &frame); // Reads in an image frame, resizes it, and calls the appropriate detection function
	cv::Point               operator>>(cv::Mat &frame); // Sets the " >> " operator to read in a frame to the function "getFrameAndDetect()"
	cv::VideoCapture*       setVideoCapture(cv::VideoCapture &videoCapture); // Sets up the webcam video object
	cv::CascadeClassifier*  setFaceCascade(const string cascadeFilePath); // Sets up the cascade classifier file object
	int                 setResizedWidth(const int width); // Sets the variable for the resized frame width
	cv::Rect                face() const;  // A rectangle which outlines the face in the image frame
	cv::Point               facePosition() const; // Set the center of the face rectangle
	double              setTemplateMatchingMaxDuration(const double s); // Sets a timer for how long to try template matching

private:
	static const double     TICK_FREQUENCY; // Created here so it can be accessed by many functions

	cv::VideoCapture*       m_videoCapture = NULL; // The standard OpenCV Webcam video capturing class
	cv::CascadeClassifier*  m_faceCascade = NULL; // The Facial Classification file 
	std::vector<cv::Rect>		m_allFaces; // All faces that were found in the "detectMultiscale" search
	cv::Rect                m_trackedFace; // The chosen face to be tracked (currently set as the largest found face
	cv::Rect                m_faceRoi; // A smaller region of interest around the previously detected face if one was detected in the last frame
	cv::Mat                 m_faceTemplate; // A rectangular of pixels as a face template that is a small region in the middle of the face
	cv::Mat                 m_matchingResult; // The resulting rectangle of pixels (best match) from template matching
	bool                m_templateMatchingRunning = false; // Check if we are running the template matching
	int64               m_templateMatchingStartTime = 0; //$$$ These two are used to measure how long template matching is running
	int64               m_templateMatchingCurrentTime = 0; //$$$ and to stop it after a set duration specified by "m_templateMatchingMaxDuration"
	bool                m_foundFace = false; // Check to see if we found a face in the previous frame
	double              m_scale; // Aspect Ratio to be maintained
	int                 m_resizedWidth = 400; // Downsize the input frame for speed
	cv::Point               m_facePosition; // The center of the face rectangle
	double              m_templateMatchingMaxDuration = 1; // Sets a timer for how long to run template matching

	cv::Rect    doubleRectSize(const cv::Rect &inputRect, const cv::Rect &frameSize) const; // The ROI is double the previously detected face size
	cv::Rect    biggestFace(std::vector<cv::Rect> &faces) const; // From all the detected faces, grab the biggest (usually the most likely) one
	cv::Point   centerOfRect(const cv::Rect &rect) const; // The center point of the detected face box
	cv::Mat     getFaceTemplate(const cv::Mat &frame, cv::Rect face); // Create a face template to be used for template matching
	void    detectFaceAllSizes(const cv::Mat &frame); // Detect faces using cascades with "detectMultiscale" in the whole image
	void    detectFaceAroundRoi(const cv::Mat &frame); // Detect faces using cascades with "detectMultiscale" in the ROI
	void    detectFacesTemplateMatching(const cv::Mat &frame); // Detect faces using template matching where the template is from a face of the previous frame
};