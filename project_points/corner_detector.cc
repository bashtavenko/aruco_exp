#include "project_points/corner_detector.h"

namespace aruco {

std::vector<Correspondence> DetectCorners(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points) {
  cv::Mat thresholded;
  std::vector<std::vector<cv::Point>> contours;
  return DetectCorners(image, dictionary, object_points, thresholded, contours);
}

std::vector<Correspondence> DetectCorners(
    const cv::Mat& image, const cv::aruco::Dictionary& dictionary,
    const std::vector<ObjectPoint>& object_points, cv::Mat& thresholded,
    std::vector<std::vector<cv::Point>>& contours) {
  std::vector<Correspondence> correspondences;

  // Preprocessing
  int64 start = cv::getTickCount();
  cv::Mat gray;
  cv::Mat blurred;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);  // Noise suppression

  // Thresholding
  cv::adaptiveThreshold(blurred, thresholded, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        11, 2);

  // Morphology
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(thresholded, thresholded, kernel);

  // Find the largest contours
  std::vector<cv::Point> largest_contour;
  cv::findContours(thresholded, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  double max_area = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > max_area) {
      max_area = area;
      largest_contour = contours[i];
    }
  }

  // This is not ideal
  // cv::Rect boundingBox = cv::boundingRect(largest_contour);
  // std::vector<cv::Point2f> points(4);
  // points[0] = cv::Point2f(boundingBox.x, boundingBox.y);
  // points[1] = cv::Point2f(boundingBox.x + boundingBox.width, boundingBox.y);
  // points[2] = cv::Point2f(boundingBox.x + boundingBox.width,
  //                         boundingBox.y + boundingBox.height);
  // points[3] = cv::Point2f(boundingBox.x, boundingBox.y + boundingBox.height);
  // for (const auto& point : points) {
  //   correspondences.emplace_back(Correspondence{.image_point = point});
  // }

  //Simplifies contour into a polygon with fewer vertices
  //while retaining its overall shape.
  std::vector<cv::Point2f> corners(4);
  cv::approxPolyDP(/*curve=*/largest_contour,
                   /*approxCurve=*/corners, /*epsilon=*/
                   0.02 * cv::arcLength(largest_contour,
                                        /*closed=*/true),
                   /*closed=*/true);

  for (const auto& corner : corners) {
    correspondences.emplace_back(Correspondence{.image_point = corner});
  }

  // BALOONEY
  // std::vector<cv::Point> approx;
  // double epsilon = 0.02 * cv::arcLength(largest_contour, true);  // Start with 2%
  // cv::approxPolyDP(largest_contour, approx, epsilon, true);
  //
  // // If still more than 4 points, increase epsilon gradually
  // while (approx.size() > 4 && epsilon < 0.1 * cv::arcLength(largest_contour, true)) {
  //   epsilon += 0.005 * cv::arcLength(largest_contour, true);
  //   cv::approxPolyDP(largest_contour, approx, epsilon, true);
  // }
  //
  // std::vector<cv::Point2f> corners(4);
  // for (int i = 0; i < std::min(4, (int)approx.size()); ++i) {
  //   corners[i] = cv::Point2f(approx[i].x, approx[i].y);
  // }
  //
  // for (const auto& corner : corners) {
  //   correspondences.emplace_back(Correspondence{.image_point = corner});
  // }


  // Take first 4 points
  // for (int32_t i = 0; i < std::min(static_cast<int32_t>(corners.size()), 4);
  //      ++i) {
  //   detected_object_points[i + 1] = cv::Point(corners[i].x, corners[i].y);
  //      }
  return correspondences;
};

}  // namespace aruco
