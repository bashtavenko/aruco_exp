#include <filesystem>
#include <unordered_set>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/aruco_dictionary.hpp"
#include "project_points/highgui_utils.h"
#include "project_points/projection.h"
#include "project_points/proto_utils.h"
#include "status_macros.h"

ABSL_FLAG(std::string, image_or_video_path, "testdata/frame_0.jpg",
          "Image or video input path");

ABSL_FLAG(std::string, calibration_path, "testdata/pixel_6a_calibration.txtpb",
          "Intrinsic camera calibration");

ABSL_FLAG(std::string, output_video_path, "", "Output of projection");

const cv::aruco::Dictionary kDictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

const std::vector<cv::Point3f> source_object_points = {
    cv::Point3f(0, 0, 0), cv::Point3f(320, 0, 0), cv::Point3f(320, 250, 0),
    cv::Point3f(0, 250, 0)};

absl::Status ProcessImage(const cv::Mat& image,
                          const aruco::IntrinsicCalibration& calibration) {
  const std::unordered_map<int32_t, cv::Point> detected_points =
      aruco::DetectArucoPoints(image, kDictionary);
  const std::vector<cv::Scalar> corner_colors = {
      aruco::kMAGENTA, aruco::kCYAN, aruco::kYELLOW, aruco::kORANGE};
  for (int i = 1; i <= 4; ++i) {
    if (detected_points.contains(i)) {
      aruco::DrawCircle(image, detected_points.at(i), corner_colors[i - 1]);
    }
  }
  if (detected_points.size() != 4) return absl::OkStatus();

  std::vector<cv::Point3f> target_source_points = {cv::Point3f(110, 100, 0)};
  std::vector<cv::Point2f> source_image_points;
  for (int i = 1; i <= 4; ++i) {
    source_image_points.emplace_back(detected_points.at(i));
  }
  auto result = aruco::ProjectPoints(calibration, source_object_points,
                                     source_image_points, target_source_points);
  if (!result.ok()) {
    LOG(WARNING) << "Failed to ProjectPoints";
  }
  for (const cv::Point2f& point : result.value()) {
    aruco::DrawCircle(image, point, aruco::kGREEN, /*size=*/50);
  }

  return absl::OkStatus();
}

absl::Status RunImage(const aruco::IntrinsicCalibration& calibration) {
  const cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_or_video_path));
  if (image.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to open image '%s'", absl::GetFlag(FLAGS_image_or_video_path)));
  }
  RETURN_IF_ERROR(ProcessImage(image, calibration));

  constexpr absl::string_view kWindow = "Detection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kWindow.data(), image);
  cv::waitKey(0);

  return absl::OkStatus();
}

absl::Status RunVideo(const aruco::IntrinsicCalibration& calibration) {
  cv::VideoCapture cap(absl::GetFlag(FLAGS_image_or_video_path));
  if (!cap.isOpened()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to open video '%s'", absl::GetFlag(FLAGS_image_or_video_path)));
  }

  // Get input video properties
  const int32_t frame_width =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int32_t frame_height =
      static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 30.0;  // Default fallback

  cv::VideoWriter writer;
  if (!absl::GetFlag(FLAGS_output_video_path).empty()) {
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    bool is_color = true;
    if (!writer.open(absl::GetFlag(FLAGS_output_video_path), fourcc, fps,
                     cv::Size(frame_width, frame_height), is_color)) {
      LOG(ERROR) << "Failed to open output video";
    }
  }
  cv::Mat frame;
  constexpr absl::string_view kWindow = "Projection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);

  int32_t frame_count = 0;
  int64_t total_processing_ticks = 0;
  for (;;) {
    if (!cap.read(frame)) {
      break;  // End of video or read error
    }

    if (frame.empty()) {
      break;  // Safety check
    }

    ++frame_count;
    int64_t start_ticks = cv::getTickCount();
    auto status = ProcessImage(frame, calibration);
    const int64_t end_ticks = cv::getTickCount();

    if (!status.ok()) {
      LOG(ERROR) << "Failed to process frame";
      continue;
    }
    total_processing_ticks += (end_ticks - start_ticks);

    if (writer.isOpened()) writer.write(frame);
    cv::imshow(kWindow.data(), frame);

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

absl::Status Run() {
  using enum aruco::FileType;
  const std::string file_path = absl::GetFlag(FLAGS_image_or_video_path);
  const aruco::FileType file_type = aruco::GetFileType(file_path);

  ASSIGN_OR_RETURN(auto proto, aruco::LoadIntrinsicFromTextProtoFile(
                                   absl::GetFlag(FLAGS_calibration_path)));
  aruco::IntrinsicCalibration calibration =
      aruco::ConvertIntrinsicCalibrationFromProto(proto);

  switch (file_type) {
    case kImage: {
      cv::Mat image = cv::imread(file_path);
      if (image.empty()) {
        return absl::InvalidArgumentError("Failed to load image: " + file_path);
      }
      RETURN_IF_ERROR(RunImage(calibration));
      break;
    }
    case kVideo: {
      cv::VideoCapture cap(file_path);
      if (!cap.isOpened()) {
        return absl::InvalidArgumentError("Failed to open video: " + file_path);
      }
      RETURN_IF_ERROR(RunVideo(calibration));
      break;
    }
    case kUnknown:
      return absl::InvalidArgumentError("Unsupported file type: " + file_path);
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
