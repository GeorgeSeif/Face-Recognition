#include <dlib/opencv.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2\face\facerec.hpp"
#include "opencv2\face.hpp"
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2\optflow.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_io.h>
#include <opencv2/core/ocl.hpp>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <fstream> // For files
#include <sstream> // For strings

using namespace dlib;
using namespace cv::face;
using namespace std;

#include "Helpers.h"
#include "trainModel.h"
#include "TanTriggs.h"
#include "preProcessing.h"

void trainModel(cv::Ptr<FaceRecognizer> model, cv::CascadeClassifier frontDetector, shape_predictor pose_model, std::vector<cv::Mat> images, std::vector<int> labels){

	std::vector<cv::Mat> PreProcessed_Faces; // Faces to be trained
	std::vector<int> newLabels; // Labels for the training faces

	// Used to shrink the images to a reasonable size for speed
	int totalRows = 0, totalCols = 0;
	cv::Size faceSize;

	for (int i = 0; i < images.size(); i++){
		// If reading in an image failed, skip the current loop iteration
		if (images[i].empty()) {
			continue;
		}

		// Downscale frame to the specified width of "m_resizedWidth" while maintaing the aspect ratio
		// This is done for speed purposes
		double scale = (double)min(400, images[i].cols) / images[i].cols;
		cv::Size resizedFrameSize = cv::Size((int)(scale*images[i].cols), (int)(scale*images[i].rows));
		cv::resize(images[i], images[i], resizedFrameSize);
	}

	for (int i = 0; i < images.size(); i++){

		if (images[i].rows <= 0 | images[i].cols <= 0) {
			continue;
		}

		std::vector< cv::Rect_<int> > Training;

		// Preblur to reduce noise
		GaussianBlur(images[i], images[i], cv::Size(5, 5), 1, 1);

		// Minimum face size is 1/5th of screen height
		// Maximum face size is 2/3rds of screen height
		frontDetector.detectMultiScale(images[i], Training, 1.1, 3, 0,
			Size(images[i].rows / 5, images[i].rows / 5),
			Size(images[i].rows * 2 / 3, images[i].rows * 2 / 3));

		for (int j = 0; j < Training.size(); j++) {

			// Process face by face:
			cv::Rect Training_j = Training[j];
			if (!(0 <= Training_j.x && 0 <= Training_j.width && Training_j.x + Training_j.width <= images[i].cols
				&& 0 <= Training_j.y && 0 <= Training_j.y && Training_j.y + Training_j.height <= images[i].rows)){
				continue;
			}

			// Crop the face from the image. 
			cv::Mat TrainingFace = images[i](Training_j);

			cv_image<unsigned char> dlibImg(TrainingFace);

			// Find the pose of each face
			std::vector<full_object_detection> shapes;
			dlib::rectangle face_rect;
			//for (unsigned long i = 0; i < Training.size(); ++i){
			if (Training_j.width != 0 || Training_j.height != 0 /*faces.size() != 0*/){
				face_rect = openCVRectToDlib(Training_j);
				full_object_detection shape = pose_model(dlibImg, face_rect);
				shapes.push_back(shape);

			}
			//}

			// Get the best UNALIGNED facial landmarks points to crop out the face
			// These points should be tight enough to exclude irrelevant 
			// background stuff that can otherwise throw off the recognition,
			// but large enough to keep all the important distinguishing facial features. 
			dlib::point top = shapes[0].part(17);
			dlib::point left = shapes[0].part(4);
			dlib::point bottom = shapes[0].part(11);
			dlib::point right = shapes[0].part(12);

			for (int x = 1; x <= 67; x++){

				if (shapes[0].part(x).y() < top.y()){
					top = shapes[0].part(x);
				}
				if (shapes[0].part(x).x() < left.x()){
					left = shapes[0].part(x);
				}
				if (shapes[0].part(x).x() > right.x()){
					right = shapes[0].part(x);
				}

			}

			// Find the rotation of the face so that we can realign it,
			// such that the two eye are horizontal
			dlib::point leftEye = shapes[0].part(36);
			dlib::point rightEye = shapes[0].part(45);
			dlib::point dlibeyeCenter = shapes[0].part(27);

			cv::Point2i eyeCenter(dlibeyeCenter.x(), dlibeyeCenter.y());

			// Get the angle between the 2 eyes.
			double dy = (rightEye.y() - leftEye.y());
			double dx = (rightEye.x() - leftEye.x());
			double len = sqrt(dx*dx + dy*dy);
			// Convert Radians to Degrees.
			double angle = atan2(dy, dx) * 180.0 / CV_PI;

			cv::Mat rot_mat(2, 3, CV_32FC1);
			cv::Mat warp_mat(2, 3, CV_32FC1);

			rot_mat = getRotationMatrix2D(eyeCenter, angle, 1);

			// Find the cropping points for the ALIGNED image
			// This is done by transforming the points by the angle found above
			dlib::point topAlign((long)top.x()*rot_mat.at<double>(0, 0) + (long)top.y()*rot_mat.at<double>(0, 1) + rot_mat.at<double>(0, 2),
				(long)top.x()*rot_mat.at<double>(1, 0) + (long)top.y()*rot_mat.at<double>(1, 1) + rot_mat.at<double>(1, 2));
			dlib::point leftAlign((long)left.x()*rot_mat.at<double>(0, 0) + (long)left.y()*rot_mat.at<double>(0, 1) + rot_mat.at<double>(0, 2),
				(long)left.x()*rot_mat.at<double>(1, 0) + (long)left.y()*rot_mat.at<double>(1, 1) + rot_mat.at<double>(1, 2));
			dlib::point bottomAlign((long)bottom.x()*rot_mat.at<double>(0, 0) + (long)bottom.y()*rot_mat.at<double>(0, 1) + rot_mat.at<double>(0, 2),
				(long)bottom.x()*rot_mat.at<double>(1, 0) + (long)bottom.y()*rot_mat.at<double>(1, 1) + rot_mat.at<double>(1, 2));
			dlib::point rightAlign((long)right.x()*rot_mat.at<double>(0, 0) + (long)right.y()*rot_mat.at<double>(0, 1) + rot_mat.at<double>(0, 2),
				(long)right.x()*rot_mat.at<double>(1, 0) + (long)right.y()*rot_mat.at<double>(1, 1) + rot_mat.at<double>(1, 2));

			// We can also extract copies of each face that are cropped, rotated upright,
			// and scaled to a standard size as shown here:
			dlib::array<array2d<unsigned char> > box_chips;
			extract_image_chips(dlibImg, get_face_chip_details(shapes), box_chips);

			dlib::rectangle faceRect;

			faceRect = dlib::rectangle((long)leftAlign.x(), (long)topAlign.y(), (long)rightAlign.x(), (long)bottomAlign.y());

			dlib::matrix<unsigned char> dlibFace = tile_images(box_chips);
			cv::Mat cvFace = toMat(dlibFace);

			cv::Rect OpenCVBox = dlibRectangleToOpenCV(faceRect);
			cv::Mat temp2;
			temp2 = images[i].clone(); // Don't touch the memory of the original image

			// Align according to the angle of the eyes
			warpAffine(temp2, temp2, rot_mat, temp2.size());

			// Crop according to the transformed landmark cropping points
			// The points were transformed according to the angle of the eyes
			cv::Mat cvImg = temp2(OpenCVBox);

			// Preprocess the faces using the Tan & Triggs preprocessing method
			//cvImg = tan_triggs_preprocessing(cvImg);

			PreProcess processor(cvImg);
			cvImg = processor.Output();

			newLabels.push_back(labels[i]);

			// Stack up the preprocessed images
			PreProcessed_Faces.push_back(cvImg);

			// Used to determine the final image size for training
			totalCols += cvImg.cols;
			totalRows += cvImg.rows;
		}

	}
	cout << "DONE LOADING\n";
	// THIS RESIZING IS DONE OUTSIDE THE LOOP AND PREPROCESSING CLASS BECAUSE WE NEED
	// TO FIRST COMPUTE THE AVERAGE FACE SIZE TO MINIMIZE THE ERROR RESULTING FROM RESIZING 
	// AND CHANGING ASPECT RATIO
	faceSize = cv::Size(totalCols / newLabels.size(), totalRows / newLabels.size());
	for (int i = 0; i < PreProcessed_Faces.size(); i++){
		if (PreProcessed_Faces[i].cols > 0 & PreProcessed_Faces[i].rows > 0){
			cv::resize(PreProcessed_Faces[i], PreProcessed_Faces[i], Size(faceSize));
		}
	}
	cout << "DONE RESIZING\n";

	// Create a FaceRecognizer and train it on the given images:
	// i.e here we will train the image model which holds the templates of all the faces
	model->update(PreProcessed_Faces, newLabels);
	cout << "DONE TRAINING\n";
}
