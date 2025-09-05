// Show library corner detector with a given image.
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
#include "project_points/corner_detector.h"
#include "project_points/highgui_utils.h"
#include "status_macros.h"

ABSL_FLAG(std::string, image_path, "testdata/corners/plastic_1.jpg",
          "Image that may have Aruco and tray");

absl::Status Run() {
  cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_path));
  if (image.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to load image '%s'", absl::GetFlag(FLAGS_image_path)));
  }

  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  const auto detectorParams = cv::aruco::DetectorParameters();
  const cv::aruco::ArucoDetector detector(dictionary, detectorParams);

  std::vector<cv::Point> corners = aruco::DetectCorners(image, detector);
  if (corners.size() < 3)
    return absl::InvalidArgumentError("Failed to detect all four corners");

  const std::vector<cv::Scalar> corner_colors = {
      aruco::kMAGENTA, aruco::kCYAN, aruco::kYELLOW, aruco::kORANGE};

  for (size_t i = 0; i < corners.size(); ++i) {
    aruco::DrawCircle(image, corners[i], corner_colors[i]);
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