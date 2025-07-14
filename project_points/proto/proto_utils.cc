#include "proto_utils.h"
#include <google/protobuf/text_format.h>
#include <filesystem>
#include <fstream>

namespace aruco_exp {

absl::StatusOr<aruco_exp::proto::IntrinsicCalibration>
LoadIntrinsicFromTextProtoFile(absl::string_view file_path) {
  std::ifstream file(file_path.data());
  std::string text_proto((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
  if (text_proto.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No file_path - ", file_path));
  }
  aruco_exp::proto::IntrinsicCalibration proto;
  if (!google::protobuf::TextFormat::ParseFromString(text_proto.data(),
                                                     &proto)) {
    return absl::InternalError("Failed to parse proto message");
  }
  return proto;
}

}  // namespace aruco_exp
