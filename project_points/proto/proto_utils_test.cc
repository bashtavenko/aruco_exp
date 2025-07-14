#include "project_points/proto/proto_utils.h"
#include "absl/status/status_matchers.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "protobuf-matchers/protocol-buffer-matchers.h"
#include "tools/cpp/runfiles/runfiles.h"

namespace aruco_exp {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::bazel::tools::cpp::runfiles::Runfiles;
using ::protobuf_matchers::EqualsProto;

TEST(LoadFromTextProto, Works) {
  const Runfiles* files = Runfiles::CreateForTest();
  const std::string text_proto_file_path =
      files->Rlocation("_main/testdata/pixel_6a_calibration.txtpb");
  auto camera_matrix = LoadIntrinsicFromTextProtoFile(text_proto_file_path);

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
}  // namespace
}  // namespace aruco_exp