#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "third_party/opencv/include/opencv2/core.hpp"
#include "third_party/opencv/include/opencv2/highgui/highgui.hpp"
#include "third_party/opencv/include/opencv2/objdetect/aruco_detector.hpp"
#include "third_party/opencv/include/opencv2/objdetect/aruco_dictionary.hpp"

absl::Status Run() {
  cv::Mat marker_image;

  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  cv::aruco::generateImageMarker(dictionary, 4, 100, marker_image, 1);
  cv::imshow("Marker", marker_image);
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