// Very basic interactive corner detection without Aruco
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "project_points/highgui_utils.h"
#include "status_macros.h"

ABSL_FLAG(std::string, image_path, "testdata/corners/plastic_1.jpg",
          "Image that may have Aruco and tray");

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

std::optional<cv::Point2f> DetectAndShowAruco(
    const cv::Mat& image, const cv::aruco::ArucoDetector& detector) {
  std::vector<int32_t> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  detector.detectMarkers(image, corners, ids, cv::noArray());
  if (ids.size() != 1) return {};
  cv::aruco::drawDetectedMarkers(image, corners, ids);

  return {};
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
  std::optional<cv::Point2f> corner = DetectAndShowAruco(image, detector);

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

  // Find the largest contours
  std::vector<std::vector<cv::Point>> contours;
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

  // Show contours
  thresholded = cv::Scalar::all(0);
  cv::drawContours(thresholded, contours, -1, cv::Scalar::all(255));
  constexpr absl::string_view kContours = "Contours";
  cv::namedWindow(kContours.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kContours.data(), thresholded);

  // Simplifies contour into a polygon with fewer vertices
  // while retaining its overall shape.
  std::vector<cv::Point2f> contour_points(4);
  cv::approxPolyDP(/*curve=*/largest_contour,
                   /*approxCurve=*/contour_points, /*epsilon=*/
                   0.01 * cv::arcLength(largest_contour,
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