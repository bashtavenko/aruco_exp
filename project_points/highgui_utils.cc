#include "project_points/highgui_utils.h"
#include <filesystem>
#include <unordered_set>

namespace aruco {

FileType GetFileType(absl::string_view file_path) {
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
                const cv::Scalar& color, int32_t size) {
  const int image_size = std::min(image.rows, image.cols);
  const int radius = std::max(2, image_size / size);
  cv::circle(image, point, radius, color, cv::FILLED);
}


}  // namespace aruco
