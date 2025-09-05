#ifndef ARUCO_EXP_CORNER_DETECTOR_H
#define ARUCO_EXP_CORNER_DETECTOR_H
#include <vector>
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "opencv2/opencv.hpp"
#include "project_points/projection.h"

namespace aruco {

// Detects corners with one Aruco marker outside:
//   x   (10,10)   (50,10)
//       o-------o
//       |       |
//       |       |
//       o-------o  (50, 100)
//      (10,100)
// Returns either 4 points or zero.
// If 4 points are returned, they numbered in clockwise order
// starting from the aruco marker.
std::vector<cv::Point> DetectCorners(const cv::Mat& image,
  const cv::aruco::ArucoDetector& detector);

}  // namespace aruco

#endif  // ARUCO_EXP_CORNER_DETECTOR_H
