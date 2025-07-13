#include "projection.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"

namespace aruco {

absl::StatusOr<std::vector<cv::Point2f>> ProjectPoints(
    const cv::Mat& camera_matrix, const cv::Mat& distortion_params,
    const std::vector<cv::Point3f>& source_object_points,
    const std::vector<cv::Point2f>& source_image_points,
    const std::vector<cv::Point3f>& target_object_points) {

  cv::Mat rvec;
  cv::Mat tvec;
  auto result = cv::solvePnP(source_object_points, source_image_points,
                             camera_matrix, distortion_params, rvec, tvec);

  // auto result = cv::solvePnPRansac(source_object_points, source_image_points,
  //                            camera_matrix, distortion_params, rvec, tvec);

  if (!result) {
    return absl::InternalError("Failed to recover camera pose.");
  }

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(target_object_points, rvec, tvec,
    camera_matrix, distortion_params, image_points);

  return image_points;
}

std::unordered_map<int32_t, cv::Point> DetectArucoPoints(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary) {
  std::unordered_map<int32_t, cv::Point> detected_object_points;

  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::aruco::DetectorParameters detectorParams =
      cv::aruco::DetectorParameters();
  cv::aruco::ArucoDetector detector(dictionary, detectorParams);

  std::vector<int32_t> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;

  detector.detectMarkers(image, corners, ids, rejected);
  for (int32_t i = 0; i < static_cast<int32_t>(corners.size()); ++i) {
    int32_t marker_id = ids[i];
    cv::Rect bbox = cv::boundingRect(corners[i]);
    cv::Point center = (bbox.tl() + bbox.br()) / 2;
    detected_object_points[marker_id] = cv::Point(center.x, center.y);
  }
  return detected_object_points;
}

}  // namespace aruco