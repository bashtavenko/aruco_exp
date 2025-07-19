#include "projection.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"

namespace aruco {

absl::StatusOr<std::vector<cv::Point2f>> ProjectPoints(
    const IntrinsicCalibration& calibration,
    const std::vector<cv::Point3f>& source_object_points,
    const std::vector<cv::Point2f>& source_image_points,
    const std::vector<cv::Point3f>& target_object_points) {
  cv::Mat rvec;
  cv::Mat tvec;
  auto result = cv::solvePnP(source_object_points, source_image_points,
                             calibration.camera_matrix,
                             calibration.distortion_params, rvec, tvec);
  if (!result) {
    return absl::InternalError("Failed to recover camera pose.");
  }

  std::vector<cv::Point2f> image_points;
  cv::projectPoints(target_object_points, rvec, tvec, calibration.camera_matrix,
                    calibration.distortion_params, image_points);

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

std::unordered_map<int32_t, cv::Point> DetectCorners(const cv::Mat& image) {
  std::unordered_map<int32_t, cv::Point> detected_object_points;

  // Preprocessing
  int64 start = cv::getTickCount();
  cv::Mat gray;
  cv::Mat blurred;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);  // Noise suppression

  // Thresholding
  cv::Mat thresholded;
  cv::adaptiveThreshold(blurred, thresholded, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        11, 2);

  // Morphology
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(thresholded, thresholded, kernel);

  // Find the largest contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresholded, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  double max_area = 0;
  std::vector<cv::Point> largest_contour;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > max_area) {
      max_area = area;
      largest_contour = contours[i];
    }
  }

  // Simplifies contour into a polygon with fewer vertices
  // while retaining its overall shape.
  std::vector<cv::Point2f> corners(4);
  cv::approxPolyDP(/*curve=*/largest_contour,
                   /*approxCurve=*/corners, /*epsilon=*/
                   0.02 * cv::arcLength(largest_contour,
                                        /*closed=*/true),
                   /*closed=*/true);

  // Take first 4 points
  for (int32_t i = 0; i < std::min(static_cast<int32_t>(corners.size()), 4);
       ++i) {
    detected_object_points[i + 1] = cv::Point(corners[i].x, corners[i].y);
  }

  return detected_object_points;
}

}  // namespace aruco