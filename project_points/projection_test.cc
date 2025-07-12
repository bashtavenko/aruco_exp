#include "projection.h"
#include <vector>
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"
#include "tools/cpp/runfiles/runfiles.h"

namespace aruco {
namespace {

using ::bazel::tools::cpp::runfiles::Runfiles;

TEST(ArucoDetection, Works) {
  const Runfiles* files = Runfiles::CreateForTest();
  const cv::Mat image =
      cv::imread(files->Rlocation("_main/testdata/frame_0.jpg"));
  ASSERT_FALSE(image.empty());

  std::unordered_map<int32_t, cv::Point> results = DetectArucoPoints(
      image, cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250));
  // TODO: Validate map
  ASSERT_THAT(results, testing::SizeIs(4));
}

TEST(Projection, Works) {
  cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
  camera_matrix.at<double>(0, 0) = 1419.35339;  // fx
  camera_matrix.at<double>(0, 2) = 574.24585;   // cx
  camera_matrix.at<double>(1, 1) = 1424.77661;  // fy
  camera_matrix.at<double>(1, 2) = 953.413879;  // cy
  camera_matrix.at<double>(2, 2) = 1.;
  cv::Mat distortion_parameters = cv::Mat::zeros(1, 5, CV_64FC1);

  std::vector<cv::Point2f> source_image_points = {
      cv::Point2f(430, 149), cv::Point2f(1384, 167), cv::Point2f(1381, 877),
      cv::Point2f(423, 873)};
  std::vector<cv::Point3f> source_object_points = {
    cv::Point3f(0, 0, 0), cv::Point3f(320, 0, 0), cv::Point3f(320, 250, 0),
    cv::Point3f(0, 250, 0)};

  std::vector<cv::Point3f> target_source_points = {cv::Point3f(110, 100, 0)};

  auto result = ProjectPoints(camera_matrix, distortion_parameters,
    source_object_points, source_image_points, target_source_points);
  ASSERT_TRUE(result.ok());
  std::vector<cv::Point2f> image_points = result.value();
  EXPECT_THAT(image_points, testing::SizeIs(1));
}

}  // namespace
}  // namespace aruco
