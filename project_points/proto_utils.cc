#include "project_points/proto_utils.h"
#include <google/protobuf/text_format.h>
#include <filesystem>
#include <fstream>

namespace aruco {

absl::StatusOr<aruco::proto::IntrinsicCalibration>
LoadIntrinsicFromTextProtoFile(absl::string_view file_path) {
  std::ifstream file(file_path.data());
  std::string text_proto((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (text_proto.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No file_path - ", file_path));
  }
  aruco::proto::IntrinsicCalibration proto;
  if (!google::protobuf::TextFormat::ParseFromString(text_proto.data(),
                                                     &proto)) {
    return absl::InternalError("Failed to parse proto message");
  }
  return proto;
}

absl::StatusOr<aruco::proto::Context> LoadContextFromProtoFile(
    absl::string_view file_path) {
  std::ifstream file(file_path.data());
  std::string text_proto((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (text_proto.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No file_path - ", file_path));
  }
  aruco::proto::Context proto;
  if (!google::protobuf::TextFormat::ParseFromString(text_proto.data(),
                                                     &proto)) {
    return absl::InternalError("Failed to parse proto message");
  }
  return proto;
}

IntrinsicCalibration ConvertIntrinsicCalibrationFromProto(
    const aruco::proto::IntrinsicCalibration& proto) {
  IntrinsicCalibration result;
  result.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
  result.camera_matrix.at<double>(0, 0) = proto.camera_matrix().fx();
  result.camera_matrix.at<double>(1, 1) = proto.camera_matrix().fy();
  result.camera_matrix.at<double>(0, 2) = proto.camera_matrix().cx();
  result.camera_matrix.at<double>(1, 2) = proto.camera_matrix().cy();

  std::vector<double> proto_params{proto.distortion_params().k1(),
                                   proto.distortion_params().k2(),
                                   proto.distortion_params().k3()};
  if (proto.distortion_params().has_k4()) {
    proto_params.push_back(proto.distortion_params().k4());
  }
  if (proto.distortion_params().has_k5()) {
    proto_params.push_back(proto.distortion_params().k5());
  }
  if (proto.distortion_params().has_p1()) {
    proto_params.push_back(proto.distortion_params().p1());
  }
  if (proto.distortion_params().has_p2()) {
    proto_params.push_back(proto.distortion_params().p2());
  }
  result.distortion_params = cv::Mat::zeros(1, proto_params.size(), CV_64FC1);
  for (size_t i = 0; i < proto_params.size(); ++i) {
    result.distortion_params.at<double>(0, i) = proto_params.at(i);
  }

  return result;
}

}  // namespace aruco
