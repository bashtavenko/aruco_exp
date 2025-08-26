#include "project_points/corner_detector.h"

namespace aruco {

std::vector<Correspondence> DetectCorners(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points) {
  std::vector<Correspondence> correspondences;

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

  for (const auto& corner : corners) {
    correspondences.emplace_back(Correspondence{.image_point = corner});
  }
  // Take first 4 points
  // for (int32_t i = 0; i < std::min(static_cast<int32_t>(corners.size()), 4);
  //      ++i) {
  //   detected_object_points[i + 1] = cv::Point(corners[i].x, corners[i].y);
  //      }
  return correspondences;
};

}  // namespace aruco
