// Miscellaneous function to and fro protos
#ifndef PROTO_UTILS_H
#define PROTO_UTILS_H
#include "project_points/proto/calibration_data.pb.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace aruco_exp {

absl::StatusOr<aruco_exp::proto::IntrinsicCalibration> LoadIntrinsicFromTextProtoFile(absl::string_view file_path);

} // namespace aruco_exp

#endif //PROTO_UTILS_H
