// Miscellaneous function to and fro protos
#ifndef PROTO_UTILS_H
#define PROTO_UTILS_H
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "project_points/projection.h"
#include "project_points/proto/calibration_data.pb.h"

namespace aruco {

absl::StatusOr<aruco::proto::IntrinsicCalibration>
LoadIntrinsicFromTextProtoFile(absl::string_view file_path);

IntrinsicCalibration ConvertIntrinsicCalibrationFromProto(
  const aruco::proto::IntrinsicCalibration& proto);

}  // namespace aruco_exp

#endif  // PROTO_UTILS_H
