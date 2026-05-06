// bazel run //visualizer:viewer_main -- --file_path=/tmp/output2.ply
// Use h keyboard key or q to quit

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"
#include "opencv2/viz.hpp"

ABSL_FLAG(std::string, file_path, "/tmp/output.ply",
          "3D point cloud file in ply, xyz, obj or stl format");
ABSL_FLAG(int32_t, width, 1000, "Viewer window width");
ABSL_FLAG(int32_t, height, 800, "Viewer window height");

absl::Status ShowPly() {
  constexpr int32_t kScale = 50;
  constexpr std::string_view kWindowName = "PLY Viewer";

  // Throws and exception if file is missing
  cv::Mat cloud = cv::viz::readCloud(absl::GetFlag(FLAGS_file_path));
  cv::viz::Viz3d window(kWindowName.data());
  window.setWindowSize(
      cv::Size(absl::GetFlag(FLAGS_width), absl::GetFlag(FLAGS_height)));
  window.setBackgroundColor(cv::viz::Color::black());
  window.showWidget("Coordinate System", cv::viz::WCoordinateSystem(kScale));
  cv::viz::WCloud cloud_widget(cloud, cv::viz::Color::white());
  window.showWidget("Point Cloud", cloud_widget);

  window.spin();
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  gflags::SetCommandLineOption("logtostderr", "1");
  if (const auto status = ShowPly(); !status.ok()) {
    LOG(ERROR) << "Failed: " << status.message();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}