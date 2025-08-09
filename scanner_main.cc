#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "project_points/highgui_utils.h"

absl::Status Run() {
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Failed to open camera."));
  }
  // Get input video properties
  const int32_t frame_width =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int32_t frame_height =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 30.0;  // Default fallback
  LOG(INFO) << absl::StreamFormat("FPS: %.0f, %.0fx%.0f", fps, frame_width,
                                  frame_height);

  const cv::aruco::Dictionary dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  const auto detectorParams = cv::aruco::DetectorParameters();
  const cv::aruco::ArucoDetector detector(dictionary, detectorParams);
  auto detect = [&detector](const cv::Mat& image) {
    std::vector<int32_t> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    detector.detectMarkers(image, corners, ids, cv::noArray());
    if (!ids.empty()) {
      cv::aruco::drawDetectedMarkers(image, corners, ids);
    }
  };

  cv::Mat frame;
  int32_t frame_count = 0;
  int64_t total_processing_ticks = 0;
  while (cap.read(frame)) {
    ++frame_count;
    const int64_t start_ticks = cv::getTickCount();
    detect(frame);
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