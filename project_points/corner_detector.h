#ifndef ARUCO_EXP_CORNER_DETECTOR_H
#define ARUCO_EXP_CORNER_DETECTOR_H
#include <vector>
#include "opencv2/opencv.hpp"
#include "project_points/projection.h"
#include "opencv2/objdetect/aruco_dictionary.hpp"

namespace aruco {

std::vector<Correspondence> DetectCorners(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points);

}  // namespace aruco

#endif  // ARUCO_EXP_CORNER_DETECTOR_H
