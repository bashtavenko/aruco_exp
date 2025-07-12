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
#include "projection.h"
#include "status_macros.h"

ABSL_FLAG(std::string, image_or_video_path, "testdata/frame_0.jpg",
          "Image or video input path");

ABSL_FLAG(std::string, output_video_path, "/tmp/projection.mp4",
          "Output of projection");

enum class FileType { kImage, kVideo, kUnknown };

const cv::aruco::Dictionary kDictionary =
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

const cv::Scalar kRED(0, 0, 255);
const cv::Scalar kGREEN(0, 255, 0);
const cv::Scalar kBLUE(255, 0, 0);
const cv::Scalar kYELLOW(0, 255, 255);
const cv::Scalar kMAGENTA(255, 0, 255);
const cv::Scalar kCYAN(255, 255, 0);
const cv::Scalar kORANGE(0, 165, 255);

inline static cv::Mat CreateCameraMatrix() {
  cv::Mat camera_matrix = cv::Mat::zeros(3, 3, CV_64FC1);
  camera_matrix.at<double>(0, 0) = 1419.35339;  // fx
  camera_matrix.at<double>(0, 2) = 574.24585;   // cx
  camera_matrix.at<double>(1, 1) = 1424.77661;  // fy
  camera_matrix.at<double>(1, 2) = 953.413879;  // cy
  camera_matrix.at<double>(2, 2) = 1.;
  return camera_matrix;
}

inline static cv::Mat CreateDistortionParameters() {
  cv::Mat distortion_parameters = cv::Mat::zeros(1, 5, CV_64FC1);
  distortion_parameters.at<double>(0, 0) = 0.130025074;     // k1
  distortion_parameters.at<double>(0, 1) = -0.593377352;    // k2
  distortion_parameters.at<double>(0, 2) = -0.00208870275;  // k3
  distortion_parameters.at<double>(0, 3) = 0.001071729;     // k4
  distortion_parameters.at<double>(0, 4) = 1.30129385;      // k5
  return distortion_parameters;
}

const std::vector<cv::Point3f> source_object_points = {
    cv::Point3f(0, 0, 0), cv::Point3f(320, 0, 0), cv::Point3f(320, 250, 0),
    cv::Point3f(0, 250, 0)};

FileType GetFileType(const std::string& file_path) {
  std::filesystem::path path(file_path);
  std::string extension = path.extension().string();
  if (!extension.empty() && extension[0] == '.') {
    extension = extension.substr(1);
  }
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 ::tolower);
  static const std::unordered_set<std::string> image_extensions = {
      "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp",
      "gif", "pbm",  "pgm", "ppm", "pxm",  "pnm"};
  static const std::unordered_set<std::string> video_extensions = {
      "mp4", "avi", "mov", "mkv",  "wmv", "flv", "webm",
      "m4v", "3gp", "mpg", "mpeg", "ts",  "mts"};

  if (image_extensions.count(extension)) {
    return FileType::kImage;
  } else if (video_extensions.count(extension)) {
    return FileType::kVideo;
  } else {
    return FileType::kUnknown;
  }
}

void DrawCircle(const cv::Mat& image, const cv::Point2f point,
                const cv::Scalar& color, int32_t size = 100) {
  const int image_size = std::min(image.rows, image.cols);
  const int radius = std::max(2, image_size / size);
  cv::circle(image, point, radius, color, cv::FILLED);
}

absl::Status ProcessImage(const cv::Mat& image, const cv::Mat& camera_matrix,
                          const cv::Mat& distortion_parameters) {
  const std::unordered_map<int32_t, cv::Point> detected_points =
      aruco::DetectArucoPoints(image, kDictionary);
  const std::vector<cv::Scalar> corner_colors = {kMAGENTA, kCYAN, kYELLOW,
                                                 kORANGE};
  for (int i = 1; i <= 4; ++i) {
    if (detected_points.contains(i)) {
      DrawCircle(image, detected_points.at(i), corner_colors[i - 1]);
    }
  }
  if (detected_points.size() != 4) return absl::OkStatus();

  std::vector<cv::Point3f> target_source_points = {cv::Point3f(110, 100, 0)};
  std::vector<cv::Point2f> source_image_points;
  for (int i = 1; i <= 4; ++i) {
    source_image_points.emplace_back(detected_points.at(i));
  }
  auto result = aruco::ProjectPoints(camera_matrix, distortion_parameters,
                                     source_object_points, source_image_points,
                                     target_source_points);
  if (!result.ok()) {
    LOG(WARNING) << "Failed to ProjectPoints";
  }
  for (const cv::Point2f& point : result.value()) {
    DrawCircle(image, point, kGREEN, /*size=*/50);
  }

  return absl::OkStatus();
}

absl::Status RunImage(const cv::Mat& camera_matrix,
                      const cv::Mat& distortion_parameters) {
  cv::Mat image = cv::imread(absl::GetFlag(FLAGS_image_or_video_path));
  if (image.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to open image '%s'", absl::GetFlag(FLAGS_image_or_video_path)));
  }
  RETURN_IF_ERROR(ProcessImage(image, camera_matrix, distortion_parameters));

  constexpr absl::string_view kWindow = "Detection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);
  cv::imshow(kWindow.data(), image);
  cv::waitKey(0);

  return absl::OkStatus();
}

absl::Status RunVideo(const cv::Mat& camera_matrix,
                      const cv::Mat& distortion_parameters) {
  cv::VideoCapture cap(absl::GetFlag(FLAGS_image_or_video_path));
  if (!cap.isOpened()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to open video '%s'", absl::GetFlag(FLAGS_image_or_video_path)));
  }

  // Get input video properties
  int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 0) fps = 30.0;  // Default fallback

  const std::string output_path = absl::GetFlag(FLAGS_output_video_path);
  cv::VideoWriter writer;
  int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  bool is_color = true;
  if (!writer.open(output_path, fourcc, fps,
                   cv::Size(frame_width, frame_height), is_color)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to open output video writer for '%s'", output_path));
  }

  cv::Mat frame;
  constexpr absl::string_view kWindow = "Projection";
  cv::namedWindow(kWindow.data(), cv::WINDOW_FREERATIO);

  for (;;) {
    if (!cap.read(frame)) {
      // Better than >> operator for error detection
      break;  // End of video or read error
    }

    if (frame.empty()) {
      break;  // Safety check
    }

    auto status = ProcessImage(frame, camera_matrix, distortion_parameters);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to process frame";
      continue;
      ;
    }

    writer.write(frame);
    cv::imshow(kWindow.data(), frame);

    int key = cv::waitKey(33) & 0xFF;  // Mask to get 8-bit value
    if (key == 27) break;              // ESC key only
  }

  return absl::OkStatus();
}

absl::Status Run() {
  const std::string file_path = absl::GetFlag(FLAGS_image_or_video_path);
  const FileType file_type = GetFileType(file_path);
  const cv::Mat camera_matrix = CreateCameraMatrix();
  const cv::Mat distortion_parameters = CreateDistortionParameters();

  switch (file_type) {
    case FileType::kImage: {
      cv::Mat image = cv::imread(file_path);
      if (image.empty()) {
        return absl::InvalidArgumentError("Failed to load image: " + file_path);
      }
      RETURN_IF_ERROR(RunImage(camera_matrix, distortion_parameters));
      break;
    }
    case FileType::kVideo: {
      cv::VideoCapture cap(file_path);
      if (!cap.isOpened()) {
        return absl::InvalidArgumentError("Failed to open video: " + file_path);
      }
      RETURN_IF_ERROR(RunVideo(camera_matrix, distortion_parameters));
      break;
    }
    case FileType::kUnknown:
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
