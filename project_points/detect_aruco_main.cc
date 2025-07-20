#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "project_points/highgui_utils.h"
#include "project_points/proto_utils.h"
#include "projection.h"
#include "status_macros.h"


ABSL_FLAG(std::string, image_path, "testdata/frame_1.jpg",
          "Image that may have Aruco tags");

ABSL_FLAG(std::string, detector_type, "aruco",
          "Type of detector. aruco or corners.");

absl::Status DetectArucoRun(const cv::Mat& image) {
  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

  LOG(INFO) << "Image size: " << image.size;
  const std::unordered_map<int32_t, cv::Point> detected_points =
      aruco::DetectArucoPoints(image, dictionary);
  for (const auto& [id, point] : detected_points) {
    LOG(INFO) << id << " " << point.x << " " << point.y;
  }

  const std::vector<cv::Scalar> corner_colors = {
      aruco::kMAGENTA, aruco::kCYAN, aruco::kYELLOW, aruco::kORANGE};
  for (int i = 1; i <= 4; ++i) {
    if (detected_points.contains(i)) {
      aruco::DrawCircle(image, detected_points.at(i), corner_colors[i - 1]);
    }
  }

  if (!detected_points.empty()) {
    constexpr absl::string_view kWindow = "Detection";
    cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
    cv::imshow(kWindow.data(), image);
    cv::waitKey(0);
  }

  return absl::OkStatus();
}

absl::Status DetectCorners(const cv::Mat& image) {
  std::unordered_map<int32_t, cv::Point> detected_points =
      aruco::DetectCorners(image);

  if (detected_points.empty()) return absl::OkStatus();

  for (const auto& [id, point] : detected_points) {
    LOG(INFO) << id << " " << point.x << " " << point.y;
  }

  const std::vector<cv::Scalar> corner_colors = {
      aruco::kMAGENTA, aruco::kCYAN, aruco::kYELLOW, aruco::kORANGE};
  for (int i = 1; i <= 4; ++i) {
    if (detected_points.contains(i)) {
      aruco::DrawCircle(image, detected_points.at(i), corner_colors[i - 1]);
    }
  }
  constexpr absl::string_view kWindow = "Detection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kWindow.data(), image);
  cv::waitKey(0);
  return absl::OkStatus();
}

absl::Status Run() {
  cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_path));
  if (image.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to load image '%s'", absl::GetFlag(FLAGS_image_path)));
  }
  if (absl::GetFlag(FLAGS_detector_type) == "aruco") {
    RETURN_IF_ERROR(DetectArucoRun(image));
  } else if (absl::GetFlag(FLAGS_detector_type) == "corners") {
    RETURN_IF_ERROR(DetectCorners(image));
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid detector type:", absl::GetFlag(FLAGS_detector_type)));
  };
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