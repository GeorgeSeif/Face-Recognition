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

#include "Helpers.h"

// This function converts the OpenCV "Rect" type to Dlib "rectangle" type
dlib::rectangle openCVRectToDlib(cv::Rect r)
{
	return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

// This function converts the Dlib "rectangle" type to OpenCV "Rect" type
cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
	return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}