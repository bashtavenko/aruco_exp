// Very basic interactive corner detection without Aruco
#include <queue>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "project_points/highgui_utils.h"
#include "status_macros.h"

ABSL_FLAG(std::string, image_path, "testdata/corners/plastic_1.jpg",
          "Image that may have Aruco and tray");

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
    double point_test =
        cv::pointPolygonTest(largest_contour[i], point.value(), true);
    if (point_test >= 0) continue;
    ;
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

std::vector<cv::Point2f> getExtremePoints(
    const std::vector<cv::Point2f>& contour) {
  std::vector<cv::Point2f> points(4);

  // Find extreme points using min/max_element
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

  points[0] = cv::Point2f(topLeft.x, topLeft.y);
  points[1] = cv::Point2f(topRight.x, topRight.y);
  points[2] = cv::Point2f(bottomRight.x, bottomRight.y);
  points[3] = cv::Point2f(bottomLeft.x, bottomLeft.y);

  return points;
}

// Returns k-largest contours by their area.
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

std::optional<cv::Point2f> DetectAndShowAruco(
    const cv::Mat& image, const cv::aruco::ArucoDetector& detector) {
  std::vector<int32_t> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  detector.detectMarkers(image, corners, ids, cv::noArray());
  if (ids.size() != 1) return {};
  cv::aruco::drawDetectedMarkers(image, corners, ids);

  const int32_t marker_id = ids[0];
  cv::Rect bbox = cv::boundingRect(corners[0]);
  const cv::Point center = (bbox.tl() + bbox.br()) / 2;
  return cv::Point(center.x, center.y);
}

absl::Status Run() {
  const cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_path));
  if (image.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to load image '%s'", absl::GetFlag(FLAGS_image_path)));
  }

  // Try to get Aruco tag
  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  const auto detectorParams = cv::aruco::DetectorParameters();
  const cv::aruco::ArucoDetector detector(dictionary, detectorParams);
  std::optional<cv::Point2f> aruco = DetectAndShowAruco(image, detector);

  // Preprocessing
  int64 start = cv::getTickCount();
  cv::Mat gray;
  cv::Mat blurred;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0);  // Noise suppression

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

  // Show the largest contours for debug
  const std::vector<cv::Scalar> contour_colors = {
      aruco::kMAGENTA, aruco::kCYAN, aruco::kYELLOW, aruco::kORANGE};
  for (size_t i = 0; i < largest_contours.size(); ++i) {
    for (const auto& point : largest_contours[i]) {
      aruco::DrawCircle(image, point, contour_colors[i % contour_colors.size()],
                        1000);
    }
  }

  // Prune contours by their distance to Aruco
  const std::vector<cv::Point>& best_contour =
      largest_contours[ClosestDistanceFromArucoIndex(largest_contours, aruco)];

  // Show all contours in thresholded image
  thresholded = cv::Scalar::all(0);
  cv::drawContours(thresholded, contours, -1, cv::Scalar::all(255));
  constexpr absl::string_view kContours = "Contours";
  cv::namedWindow(kContours.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kContours.data(), thresholded);

  // Show the best contour in the main image
  for (const auto& point : best_contour) {
    aruco::DrawCircle(image, point, aruco::kBLUE, 500);
  }

  // Simplifies contour into a polygon with fewer vertices
  // while retaining its overall shape.
  std::vector<cv::Point2f> contour_points(4);
  cv::approxPolyDP(/*curve=*/best_contour,
                   /*approxCurve=*/contour_points, /*epsilon=*/
                   0.01 * cv::arcLength(best_contour,
                                        /*closed=*/true),
                   /*closed=*/true);

  // Get the most extreme points.
  // TODO: Filter out outliers first.
  const std::vector<cv::Point2f> points = getExtremePoints(contour_points);

  // Show detection
  for (const auto& point : points) {
    aruco::DrawCircle(image, point, aruco::kRED);
  }
  constexpr absl::string_view kWindow = "Detection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kWindow.data(), image);

  cv::waitKey(0);
  cv::destroyAllWindows();

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  gflags::SetCommandLineOption("logtostderr", "1");
  if (const auto status = Run(); !status.ok()) {
    LOG(ERROR) << "Failed: " << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}