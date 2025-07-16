// Miscellaneous functions for highgui module
#ifndef HIGHGUI_UTILS_H
#define HIGHGUI_UTILS_H
#include <opencv2/opencv.hpp>
#include "absl/strings/string_view.h"

namespace aruco {

const cv::Scalar kRED(0, 0, 255);
const cv::Scalar kGREEN(0, 255, 0);
const cv::Scalar kBLUE(255, 0, 0);
const cv::Scalar kYELLOW(0, 255, 255);
const cv::Scalar kMAGENTA(255, 0, 255);
const cv::Scalar kCYAN(255, 255, 0);
const cv::Scalar kORANGE(0, 165, 255);

enum class FileType { kImage, kVideo, kUnknown };

// Given file path returns FileType based on the file path extension.
FileType GetFileType(absl::string_view file_path);

// Draws circle for the given point and color. The smaller size the bigger
// circle.
void DrawCircle(const cv::Mat& image, const cv::Point2f point,
                const cv::Scalar& color, int32_t size = 100);

}  // namespace aruco

#endif  // HIGHGUI_UTILS_H
