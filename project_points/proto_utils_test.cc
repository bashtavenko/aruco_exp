#include "project_points/proto_utils.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "protobuf-matchers/protocol-buffer-matchers.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace aruco {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::bazel::tools::cpp::runfiles::Runfiles;
using ::protobuf_matchers::EqualsProto;

MATCHER_P2(CompareMat, a, b, "") {
  cv::Mat difference;
  cv::absdiff(a, b, difference);
  return cv::sum(difference).val[0] <= arg;
}

TEST(LoadFromTextProto, Works) {
  const Runfiles* files = Runfiles::CreateForTest();
  const std::string text_proto_file_path =
      files->Rlocation("_main/testdata/pixel_6a_calibration.txtpb");

  EXPECT_THAT(LoadIntrinsicFromTextProtoFile(text_proto_file_path),
              IsOkAndHolds(EqualsProto(
                  R"pb(camera_matrix {
                         fx: 1419.35339
                         fy: 1424.77661
                         cx: 574.24585
                         cy: 953.413879
                       }
                       distortion_params {
                         k1: 0.130025074
                         k2: -0.593377352
                         k3: -0.00208870275
                         k4: 0.001071729
                         k5: 1.30129385
                       }
                       reprojection_error: 0.334963739
                  )pb")));
}

TEST(LoadFromTextProtoAndConvert, Works) {
  const Runfiles* files = Runfiles::CreateForTest();
  auto intrinsic_proto = LoadIntrinsicFromTextProtoFile(
      files->Rlocation("_main/testdata/pixel_6a_calibration.txtpb"));
  ASSERT_THAT(intrinsic_proto, IsOk());

  const auto intrinsic =
      ConvertIntrinsicCalibrationFromProto(intrinsic_proto.value());
  const cv::Mat want_camera_matrix(
      {3, 3}, {1419., 0.0, 574., 0.0, 1424., 953., 0.0, 0.0, 1.0});
  EXPECT_THAT(2.0, CompareMat(intrinsic.camera_matrix, want_camera_matrix));
  const cv::Mat want_distortion_params(
      {1, 5},
      {0.130025074, -0.593377352, -0.00208870275, 0.001071729, 1.30129385});
  EXPECT_THAT(0.1,
              CompareMat(intrinsic.distortion_params, want_distortion_params));
}
}  // namespace
}  // namespace aruco