#ifndef PROJECTION_H
#define PROJECTION_H
#include "absl/status/statusor.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
// #include "calibration_data.pb.h"
#include <unordered_map>
#include "opencv2/objdetect/aruco_dictionary.hpp"

namespace aruco {

// Detects Aruco corners in the map for the given dictionary.
// It can return 0..4 detected points
std::unordered_map<int32_t, cv::Point>DetectArucoPoints(const cv::Mat& image,
  const cv::aruco::Dictionary& dictionary);

// Projects source object points to the taget and returns image points.
absl::StatusOr<std::vector<cv::Point2f>> ProjectPoints(const cv::Mat& camera_matrix,
  const cv::Mat& distortion_params,
  const std::vector<cv::Point3f>& source_object_points,
  const std::vector<cv::Point2f>& source_image_points,
  const std::vector<cv::Point3f>& target_object_points);

}  // namespace aruco

#endif  // PROJECTION_H
