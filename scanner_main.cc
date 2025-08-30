#include "status_macros.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "project_points/highgui_utils.h"

ABSL_FLAG(std::string, image_path, "testdata/corners/plastic_1.jpg",
          "Image that may have Aruco tags. If empty tries to open camera.");

void DetectAndDrawAruco(const cv::Mat& image,
                        const cv::aruco::ArucoDetector& detector) {
  std::vector<int32_t> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  detector.detectMarkers(image, corners, ids, cv::noArray());
  if (!ids.empty()) {
    cv::aruco::drawDetectedMarkers(image, corners, ids);
  }
}

absl::Status RunVideo() {
  cv::VideoCapture cap(0);
  const int32_t frame_width =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int32_t frame_height =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 30.0;  // Default fallback
  LOG(INFO) << absl::StreamFormat("FPS: %.0f, %.0fx%.0f", fps, frame_width,
                                  frame_height);

  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  const auto detectorParams = cv::aruco::DetectorParameters();
  const cv::aruco::ArucoDetector detector(dictionary, detectorParams);

  cv::Mat frame;
  int32_t frame_count = 0;
  int64_t total_processing_ticks = 0;
  while (cap.read(frame)) {
    ++frame_count;
    const int64_t start_ticks = cv::getTickCount();
    DetectAndDrawAruco(frame, detector);
    cv::imshow("Scanner", frame);
    const int64_t end_ticks = cv::getTickCount();
    total_processing_ticks += (end_ticks - start_ticks);

    if (const int key = cv::waitKey(33) & 0xFF; key == 27)
      break;  // ESC key only
  }

  const double total_processing_time_ms =
      total_processing_ticks / cv::getTickFrequency() * 1000.0;
  const double processing_fps =
      frame_count / (total_processing_time_ms / 1000.0);
  const double mean_ms_per_frame = total_processing_time_ms / frame_count;
  LOG(INFO) << absl::StreamFormat("Mean FPS: %.0f", processing_fps);
  LOG(INFO) << absl::StreamFormat("Mean latency %.0f ms", mean_ms_per_frame);

  return absl::OkStatus();
}

absl::Status RunImage(const cv::Mat& image) {
  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  const auto detectorParams = cv::aruco::DetectorParameters();
  const cv::aruco::ArucoDetector detector(dictionary, detectorParams);

  DetectAndDrawAruco(image, detector);
  constexpr absl::string_view kWindow = "Detection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kWindow.data(), image);
  cv::waitKey(0);
  return absl::OkStatus();
}

absl::Status Run() {
  if (absl::GetFlag(FLAGS_image_path).empty()) {
    RETURN_IF_ERROR(RunVideo());
  } else {
    cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_path));
    if (image.empty()) {
      return absl::InvalidArgumentError("Image not found: " +
                                        absl::GetFlag(FLAGS_image_path));
    }
    RETURN_IF_ERROR(RunImage(image));
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