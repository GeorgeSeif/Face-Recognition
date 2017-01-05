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

#include "VideoFaceDetector.h"
#include "preProcessing.h"
#include "Helpers.h"
#include "TanTriggs.h"
#include "trainModel.h"

#include <fstream> // For files
#include <sstream> // For strings

using namespace dlib;
using namespace cv::face;
using namespace std;

#include <sys/timeb.h>
#include <time.h>

#if defined(_MSC_VER) || defined(WIN32)  || defined(_WIN32) || defined(__WIN32__) \
    || defined(WIN64)    || defined(_WIN64) || defined(__WIN64__) 
int CLOCK()
{
	return clock();
}
#endif

#if defined(unix)        || defined(__unix)      || defined(__unix__) \
    || defined(linux)       || defined(__linux)     || defined(__linux__) \
    || defined(sun)         || defined(__sun) \
    || defined(BSD)         || defined(__OpenBSD__) || defined(__NetBSD__) \
    || defined(__FreeBSD__) || defined __DragonFly__ \
    || defined(sgi)         || defined(__sgi) \
    || defined(__MACOSX__)  || defined(__APPLE__) \
    || defined(__CYGWIN__) 
int CLOCK()
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (t.tv_sec * 1000) + (t.tv_nsec*1e-6);
}
#endif

double _avgdur = 0;
int _fpsstart = 0;
double _avgfps = 0;
double _fps1sec = 0;

double avgdur(double newdur)
{
	_avgdur = 0.98*_avgdur + 0.02*newdur;
	return _avgdur;
}

double avgfps()
{
	if (CLOCK() - _fpsstart>1000)
	{
		_fpsstart = CLOCK();
		_avgfps = 0.5*_avgfps + 0.5*_fps1sec;
		_fps1sec = 0;
	}

	_fps1sec++;
	return _avgfps;
}

void read_csv(const string& filename, std::vector<Mat>& images, std::vector<int>& labels);

const String    CASCADE_FILE("C:\\George Seif\\opencv 3\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
const String    leftEyeFile("C:\\George Seif\\opencv 3\\opencv\\sources\\data\\haarcascades\\haarcascade_lefteye_2splits.xml");
const String    rightEyeFile("C:\\George Seif\\opencv 3\\opencv\\sources\\data\\haarcascades\\haarcascade_righteye_2splits.xml");
const String	fn_csv = string("C:\\George Seif\\OpenCV Test Images\\EDP Facial Recognition Images.txt");

int main()
{
	//cv::ocl::setUseOpenCL(false);
	cv::Mat::getStdAllocator();

	// Load face detection and pose estimation models.
	shape_predictor pose_model;
	deserialize("C:\\George Seif\\dlib-18.18\\shape_predictor_68_face_landmarks.dat") >> pose_model;

	// These vectors hold the images and corresponding labels:
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}

	cv::CascadeClassifier frontDetector;
	frontDetector.load(CASCADE_FILE);
	
	

	// Create a FaceRecognizer and train it on the given images:
	// i.e here we will train the image model which holds the templates of all the faces
	cv::Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	trainModel(model, frontDetector, pose_model, images, labels);
	bool faceSent = false;

	std::vector< cv::Rect_<int> > leftEyes;
	std::vector< cv::Rect_<int> > rightEyes;
	
	bool pictureSent = false;
	int pictureCount = 0;
	int frameCounter = 0;

	try
	{
		cv::VideoCapture cap(0);
		image_window win, win_faces, win_mask;

		VideoFaceDetector VideoDetector(CASCADE_FILE, cap);

		// Grab and process frames until the main window is closed by the user.
		while (!win.is_closed())
		{ 

			clock_t start = CLOCK();
			// Grab a frame
			cv::Mat color_temp;
			cv::Mat temp;
			VideoDetector >> color_temp;
			if (color_temp.rows <= 0 | color_temp.cols <= 0){
				continue;
			}
			
			cv::cvtColor(color_temp, temp, CV_BGR2GRAY);

			// Preblur to reduce noise
			GaussianBlur(temp, temp, cv::Size(5, 5), 1, 1);

			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<unsigned char> cimg(temp);
			cv_image<bgr_pixel> color_cimg(color_temp);

			// Detect the face and find the bounding box of each face.
			Rect face_i = VideoDetector.face();

			if (!(0 <= face_i.x && 0 <= face_i.width && face_i.x + face_i.width <= temp.cols
				&& 0 <= face_i.y && 0 <= face_i.y && face_i.y + face_i.height <= temp.rows)){
				continue;
			}

			// Get the facial landmarks of the face
			std::vector<full_object_detection> shapes;
			dlib::rectangle face_rect;

			if (face_i.width != 0 | face_i.height != 0){
				face_rect = openCVRectToDlib(face_i);
				full_object_detection shape = pose_model(cimg, face_rect);
				shapes.push_back(shape);
				
			}
			
			if (shapes.size() == 0) {
				continue;
			}

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
			extract_image_chips(cimg, get_face_chip_details(shapes), box_chips);

			dlib::rectangle faceRect;
	
			faceRect = dlib::rectangle((long)leftAlign.x(), (long)topAlign.y(), (long)rightAlign.x(), (long)bottomAlign.y());

			dlib::matrix<unsigned char> dlibFace = tile_images(box_chips);
			
			cv::Mat cvFace = toMat(dlibFace);
			

			cv::Rect OpenCVBox = dlibRectangleToOpenCV(faceRect); 
			cv::Mat temp2;
			temp2 = temp.clone(); // Don't touch the memory of the original image
			
			// Align according to the angle of the eyes
			warpAffine(temp2, temp2, rot_mat, temp2.size()); 
			
			// Crop according to the transformed landmark cropping points
			// The points were transformed according to the angle of the eyes
			if (!(0 <= OpenCVBox.x && 0 <= OpenCVBox.width && OpenCVBox.x + OpenCVBox.width <= temp2.cols
				&& 0 <= OpenCVBox.y && 0 <= OpenCVBox.y && OpenCVBox.y + OpenCVBox.height <= temp2.rows)){
				continue;
			}
			
			cv::Mat cropped = temp2(OpenCVBox);
			

			// Also get a cropped image in colour
			// Convert to the HSV colour space for skin detection and preblur
			Mat frameClone;
			frameClone = color_temp.clone(); // Use a clone so that we don't touch the memory where this image is saved
			cv::Mat colorCropped = frameClone(OpenCVBox);
			cvtColor(colorCropped, colorCropped, CV_BGR2HSV); // Convert to the HSV colour space for skin detection
			GaussianBlur(colorCropped, colorCropped, Size(7, 7), 1, 1); // Preblur to reduce noise

			// If the colour of the cropped image is in the HSV range for skin
			// then keep it (i.e set it to 1 in the mask, if not set to 0)
			Mat mask;
			inRange(colorCropped, Scalar(4, 20, 0), Scalar(40, 250, 255), mask);
			// Dilate the mask to get a better picture of things
			Mat out;
			int dilateSize = 2*ceil(sqrt(0.006*mask.rows*mask.cols)) + 1;
			Mat element = getStructuringElement(MORPH_RECT, Size(dilateSize, dilateSize));
			morphologyEx(mask, out, MORPH_DILATE, element);
			// If the ratio between pixels that were found to resemble skin colour and the total number 
			// of pixels is above a certain threshold, then we have detected a face. If not, then it is 
			// probably not a face.
			double pixelRatio = (double)countNonZero(out) / ((double)out.cols*(double)out.rows);
			string Ratio = format("Ratio = %f", pixelRatio);
			string Dimension = format("Size = %d", dilateSize);
			putText(out, Ratio, Point(8, 16), FONT_HERSHEY_PLAIN, 0.75, CV_RGB(255, 0, 0), 1);
			putText(out, Dimension, Point(8, 24), FONT_HERSHEY_PLAIN, 0.75, CV_RGB(255, 0, 0), 1);
			cv_image<unsigned char> dlibMask(out);
			win_mask.set_title("Mask");
			win_mask.set_image(dlibMask);

			cv::Mat cvImg = cropped;

			PreProcess processor(cvImg);
			cv::Mat finalFace = processor.Output();

			//cv::Mat finalFace = tan_triggs_preprocessing(cvImg);

			if (!(finalFace.rows > 5 & finalFace.cols > 5)){
				continue;
			}

			double Dist = 0.0;
			int ID_Label = -1;

			// Recognize the face
			model->predict(finalFace, ID_Label, Dist);
			
			cv_image<unsigned char> faceBox(cvImg);

			// Calculate the position for annotated text (make sure we don't put illegal values in there):
			int pos_x = 2;
			int pos_y = temp.rows - 5;
			// Create the text we will annotate the box with:
			string identity = format("Face Prediction = %d", ID_Label);
			// Compute the similarity metric
			double Sim = 0;
			if (Dist < 70){
				Sim = 100;
			}
			else if((70 <= Dist) & (Dist <= 170)){
				Sim = 100 - (Dist - 70);
			}
			else{
				Sim = 0;
			}
			string Similarity = format("Similarity = %.4f%%", Dist);
			string Angle = format("Rotation Angle = %.4f Degrees", angle);
			
			// Display it all on the screen
			win.set_title("Original Frames");
			win_faces.set_title("Aligned and Cropped Facial Landmarks");
			win.clear_overlay();
			win_faces.clear_overlay();
			
			if (pixelRatio > 0.65){
				putText(color_temp, identity, Point(8, 15), FONT_HERSHEY_PLAIN, 1.25, CV_RGB(255, 0, 0), 1);
				//putText(color_temp, Similarity, Point(8, 30), FONT_HERSHEY_PLAIN, 1.25, CV_RGB(255, 0, 0), 1);
				putText(color_temp, Angle, Point(8, 30), FONT_HERSHEY_PLAIN, 1.25, CV_RGB(255, 0, 0), 1);
				win.set_image(color_cimg);
				win.add_overlay(face_rect, rgb_pixel(255, 0, 0));
				win.add_overlay(render_face_detections(shapes));
				win_faces.set_image(faceBox);
				if (pictureSent == false){
					if (pictureCount >= 5){
						imwrite("Person Detected.png", color_temp);
						pictureSent = true;
					}
					pictureCount++;
				}
				frameCounter = 0;

			}
			else{
				win.set_image(color_cimg);
				// This variable counts the number of frames that 
				// have passed where a face has not been detected.
				// If a face has not been detected in a while then 
				// the person must have left. Now if anyone enters 
				// the frame, send a picture of them. 
				frameCounter++;
				if (frameCounter > 100){
					pictureSent = false;
				}
			}
			
			double dur = CLOCK() - start;
			printf("FPS = %f.\n\n", avgfps());
			
		}
	}
	catch (serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch (exception& e)
	{
		cout << e.what() << endl;
	}
}

void read_csv(const string& filename, std::vector<Mat>& images, std::vector<int>& labels) {
	char separator = ';';
	// ifstream: Stream class to read from files
	std::ifstream file(filename.c_str(), ifstream::in); // Load the given file 

	// If the file with the given name does not exist, throw an error
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}

	string line, path, classlabel;
	while (getline(file, line)) { // For each line
		stringstream liness(line); // Read the file line by line 
		getline(liness, path, separator); // Read in the image file path up to the semicolon ";" seperator
		getline(liness, classlabel); // Read the next of the line which is the image label number

		// If we were able to successfully read in the image with its label, then load them into the corresponding vectors
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}


