#include "project_points/corner_detector.h"
#include "glog/logging.h"
#include <queue>

namespace aruco {


// Returns the index of the largest contours that is closes to the given
// Aruco point (represented by the center of the Aruco marker).
size_t ClosestDistanceFromArucoIndex(
    const std::vector<std::vector<cv::Point>>& largest_contour,
    const std::optional<cv::Point2f> point) {
  CHECK(!largest_contour.empty());
  if (!point.has_value()) return largest_contour.size() - 1;

  size_t closest_contour_index = 0;
  double min_distance = std::numeric_limits<double>::max();
  bool found_valid_contour = false;

  for (size_t i = 0; i < largest_contour.size(); ++i) {
    // Check if Aruco point is inside this contour
    const double point_test =
        cv::pointPolygonTest(largest_contour[i], point.value(), true);
    if (point_test >= 0) continue;

    const double distance = std::abs(point_test);
    if (distance < min_distance) {
      min_distance = distance;
      closest_contour_index = i;
      found_valid_contour = true;
    }
  }

  if (!found_valid_contour) return largest_contour.size() - 1;
  return closest_contour_index;
}

// Given a contour returns four most extreme points
std::vector<cv::Point> getExtremePoints(
    const std::vector<cv::Point>& contour) {
  CHECK(contour.size() > 3);
  std::vector<cv::Point> points(4);

  auto topLeft = *std::min_element(contour.begin(), contour.end(),
                                   [](const cv::Point& a, const cv::Point& b) {
                                     return (a.x + a.y) < (b.x + b.y);
                                   });

  auto topRight = *std::min_element(contour.begin(), contour.end(),
                                    [](const cv::Point& a, const cv::Point& b) {
                                      return (a.y - a.x) < (b.y - b.x);
                                    });

  auto bottomRight =
      *std::max_element(contour.begin(), contour.end(),
                        [](const cv::Point& a, const cv::Point& b) {
                          return (a.x + a.y) < (b.x + b.y);
                        });

  auto bottomLeft =
      *std::max_element(contour.begin(), contour.end(),
                        [](const cv::Point& a, const cv::Point& b) {
                          return (a.y - a.x) < (b.y - b.x);
                        });

  points[0] = cv::Point(topLeft.x, topLeft.y);
  points[1] = cv::Point(topRight.x, topRight.y);
  points[2] = cv::Point(bottomRight.x, bottomRight.y);
  points[3] = cv::Point(bottomLeft.x, bottomLeft.y);

  return points;
}

// Returns k-largest contours by their area
std::vector<std::vector<cv::Point>> GetLargestContours(
    const std::vector<std::vector<cv::Point>>& contours, size_t k) {
  std::priority_queue<std::vector<cv::Point>,
                      std::vector<std::vector<cv::Point>>,
                      std::function<bool(const std::vector<cv::Point>&,
                                         const std::vector<cv::Point>&)>>
      min_heap(
          [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) >= cv::contourArea(b);
          });
  for (auto contour : contours) {
    min_heap.emplace(contour);
    if (min_heap.size() == k + 1) {
      min_heap.pop();
    }
  }
  std::vector<std::vector<cv::Point>> top_contours;
  while (!min_heap.empty()) {
    top_contours.emplace_back(min_heap.top());
    min_heap.pop();
  }
  return top_contours;
}

// Returns the center of detected Aruco marker in the given image
// and detector. Returns empty if not found.
std::optional<cv::Point> DetectAruco(
    const cv::Mat& image, const cv::aruco::ArucoDetector& detector) {
  std::vector<int32_t> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  detector.detectMarkers(image, corners, ids, cv::noArray());
  if (ids.size() != 1) return {};

  const cv::Rect bbox = cv::boundingRect(corners[0]);
  const cv::Point center = (bbox.tl() + bbox.br()) / 2;
  return cv::Point(center.x, center.y);
}





std::vector<cv::Point> DetectCorners(const cv::Mat& image,
  const cv::aruco::ArucoDetector& detector) {

  // Preprocessing
  cv::Mat gray;
  cv::Mat blurred;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0);  // Noise suppression

  // Try to get Aruco tag
  std::optional<cv::Point2f> aruco = DetectAruco(image, detector);

  // Thresholding
  cv::Mat thresholded;
  cv::adaptiveThreshold(blurred, thresholded, 255,
                        cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                        11, 2);

  // Morphology
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::dilate(thresholded, thresholded, kernel);

  // Find k-largest contour by their area
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresholded, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  std::vector<std::vector<cv::Point>> largest_contours =
      GetLargestContours(contours, 3);

  // Prune contours by their distance to Aruco
  const std::vector<cv::Point>& best_contour =
      largest_contours[ClosestDistanceFromArucoIndex(largest_contours, aruco)];

  // Simplifies contour into a polygon with fewer vertices
  // while retaining its overall shape.
  std::vector<cv::Point> contour_points(4);
  cv::approxPolyDP(/*curve=*/best_contour,
                   /*approxCurve=*/contour_points, /*epsilon=*/
                   0.01 * cv::arcLength(best_contour,
                                        /*closed=*/true),
                   /*closed=*/true);

  // Get the most extreme points.
  // TODO: Filter out outliers first.
  const std::vector<cv::Point> points = getExtremePoints(contour_points);
  return points;
}
}  // namespace aruco
