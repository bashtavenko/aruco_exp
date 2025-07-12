#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "projection.h"
#include "status_macros.h"

ABSL_FLAG(std::string, image_path, "testdata/frame_1.jpg",
          "Image that may have Aruco tags");

absl::Status Run() {
  cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_path));
  if (image.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to load image '%s'", absl::GetFlag(FLAGS_image_path)));
  }

  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  const std::unordered_map<int32_t, cv::Point> detected_points =
      aruco::DetectArucoPoints(image, dictionary);
  for (const auto& [id, point] : detected_points) {
    LOG(INFO) << id << " " << point.x << " " << point.y;
  }

  if (!detected_points.empty()) {
    constexpr absl::string_view kWindow = "Detection";
    cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
    cv::imshow(kWindow.data(), image);
    cv::waitKey(0);
  }

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