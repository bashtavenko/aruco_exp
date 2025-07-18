// Miscellaneous function to and fro protos
#ifndef PROTO_UTILS_H
#define PROTO_UTILS_H
#include <google/protobuf/text_format.h>
#include <fstream>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "project_points/projection.h"
#include "project_points/proto/calibration_data.pb.h"
#include "project_points/proto/manifest.pb.h"

namespace aruco {

template <typename ProtoType>
absl::StatusOr<ProtoType> LoadFromTextProtoFile(absl::string_view file_path) {
  static_assert(std::is_base_of_v<google::protobuf::Message, ProtoType>,
                "ProtoType must be a protobuf message type");

  std::ifstream file(file_path.data());
  if (!file.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open file: ", file_path));
  }

  std::string text_proto((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

  if (text_proto.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("Empty file: ", file_path));
  }

  ProtoType proto;
  if (!google::protobuf::TextFormat::ParseFromString(text_proto, &proto)) {
    return absl::InternalError(
        absl::StrCat("Failed to parse proto message from file: ", file_path));
  }

  return proto;
}

// Converts proto into struct
IntrinsicCalibration ConvertIntrinsicCalibrationFromProto(
    const aruco::proto::IntrinsicCalibration& proto);

// Writes proto to the text proto
template <typename ProtoType>
absl::StatusOr<std::string> WriteProtoToTextProto(ProtoType proto,
                                                  absl::string_view file_path) {
  static_assert(std::is_base_of_v<google::protobuf::Message, ProtoType>,
                "ProtoType must be a protobuf message type");
  std::string text_format;
  google::protobuf::TextFormat::PrintToString(proto, &text_format);
  std::ofstream output_file(file_path.data());
  if (!output_file) {
    return absl::InternalError(absl::StrCat("Failed writing to ", file_path));
  }
  output_file << text_format;
  output_file.close();
  return text_format;
}

std::vector<cv::Point3f> ConvertContextToObjectPoints(
    const aruco::proto::Context& proto);

std::unordered_map<std::int32_t, cv::Point3f> ConvertContextToItemPoints(
    const aruco::proto::Context& proto);

}  // namespace aruco

#endif  // PROTO_UTILS_H
