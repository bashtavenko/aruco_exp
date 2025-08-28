#ifndef ARUCO_EXP_CORNER_DETECTOR_H
#define ARUCO_EXP_CORNER_DETECTOR_H
#include <vector>
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "opencv2/opencv.hpp"
#include "project_points/projection.h"

namespace aruco {

std::vector<Correspondence> DetectCorners(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points);

std::vector<Correspondence> DetectCorners(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points,
    std::vector<std::vector<cv::Point>>& contours,
    std::vector<cv::Point>& best_contour);

}  // namespace aruco

#endif  // ARUCO_EXP_CORNER_DETECTOR_H
