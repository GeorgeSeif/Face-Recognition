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

dlib::rectangle openCVRectToDlib(cv::Rect r);
cv::Rect dlibRectangleToOpenCV(dlib::rectangle r);
