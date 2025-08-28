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

ABSL_FLAG(std::string, image_path, "testdata/corners/plastic_1_90.jpg",
          "Image that may have Aruco and tray");

absl::Status DetectCorners(const cv::Mat& image) {
  cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

  std::vector<aruco::ObjectPoint> object_points;

  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> best_contour;

  std::vector<aruco::Correspondence> result = DetectCorners(
      image, dictionary, object_points, contours, best_contour);

  if (result.empty()) return absl::OkStatus();

  for (const aruco::Correspondence& correspondence : result) {
    aruco::DrawCircle(image, correspondence.image_point, aruco::kRED);
  }

  cv::Mat thresholded;
  thresholded = cv::Scalar::all(0);
  cv::drawContours(thresholded, contours, -1, cv::Scalar::all(255));

  constexpr absl::string_view kWindow = "Detection";
  constexpr absl::string_view kContours = "Contours";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::namedWindow(kContours.data(), cv::WINDOW_FREERATIO);
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
  RETURN_IF_ERROR(DetectCorners(image));
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