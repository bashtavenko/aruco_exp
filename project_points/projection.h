#ifndef PROJECTION_H
#define PROJECTION_H
#include <unordered_map>
#include "absl/status/statusor.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
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

// Linkage of detected aruco image point and corresponding objection point;
struct Correspondence {
  int32_t id;
  cv::Point2f image_point;
  cv::Point3f object_point;
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
  cv::aruco::Dictionary dictionary;
};

// Given an image with Aruco tags, Aruco dictionary and object points,
// it returns the vector of correspondence.
// For example, there are object points [{0, 0, "1"}, {320, 0, "2"}]
// and only one Aruco marker was detected as 2, then the function returns
// [{2, {10, 2} {320, 0}}] assuming that Aruco tag 2 was detected at {10, 2}.
std::vector<Correspondence> DetectArucoPoints(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points);

// Detects corners of the biggest contour.
std::unordered_map<int32_t, cv::Point> DetectCorners(const cv::Mat& image);

// Projects source object points to the taget and returns image points.
absl::StatusOr<std::vector<cv::Point2f>> ProjectPoints(
    const IntrinsicCalibration& calibration,
    const std::vector<cv::Point3f>& source_object_points,
    const std::vector<cv::Point2f>& source_image_points,
    const std::vector<cv::Point3f>& target_object_points);

}  // namespace aruco

#endif  // PROJECTION_H
