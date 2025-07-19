#ifndef PROJECTION_H
#define PROJECTION_H
#include "absl/status/statusor.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
// #include "calibration_data.pb.h"
#include <unordered_map>
#include "opencv2/objdetect/aruco_dictionary.hpp"

namespace aruco {

struct IntrinsicCalibration {
  cv::Mat camera_matrix;
  cv::Mat distortion_params;
};

struct ObjectPoint {
  cv::Point3f point;
  std::string tag;
};

struct Item {
  int32_t id;
  std::string name;
  size_t count;
};

struct ItemObjectPoint {
  int32_t id;
  cv::Point3f object_point;
};

struct Context {
  std::vector<ObjectPoint> object_points;
  std::vector<Item> items;
  std::vector<ItemObjectPoint> item_points;
};


// Detects Aruco corners in the map for the given dictionary.
// It can return 0..4 detected points
std::unordered_map<int32_t, cv::Point>DetectArucoPoints(const cv::Mat& image,
  const cv::aruco::Dictionary& dictionary);

// Detects corners of the biggest contour.
std::unordered_map<int32_t, cv::Point>DetectCorners(const cv::Mat& image);

// Projects source object points to the taget and returns image points.
absl::StatusOr<std::vector<cv::Point2f>> ProjectPoints(const IntrinsicCalibration& calibration,
  const std::vector<cv::Point3f>& source_object_points,
  const std::vector<cv::Point2f>& source_image_points,
  const std::vector<cv::Point3f>& target_object_points);

}  // namespace aruco

#endif  // PROJECTION_H
