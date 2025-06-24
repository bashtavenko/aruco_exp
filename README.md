# OpenCV Aruco with Bazel

Building and running OpenCV core modules hermetically with Bazel is straightforward.

**MODULE.bazel**

```shell
bazel_dep(name = "rules_foreign_cc", version = "0.10.1")
bazel_dep(name = "rules_cc", version = "0.0.16")

http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

all_content = """\
filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

http_archive(
    name = "opencv",
    build_file_content = all_content,
    sha256 = "cbf47ecc336d2bff36b0dcd7d6c179a9bb59e805136af6b9670ca944aef889bd",
    strip_prefix = "opencv-4.8.0",
    urls = ["https://github.com/opencv/opencv/archive/refs/tags/4.8.0.tar.gz"],
)
```

**BUILD.bazel**

```shell
cmake(
    name = "opencv",
    cache_entries = {
        "BUILD_LIST": "calib3d,core,features2d,highgui,imgcodecs,imgproc,video,videoio",
        "WITH_FFMPEG": "ON",
        "WITH_GTK": "OFF",
        "WITH_QT": "ON",
    },
    lib_source = "@opencv//:all",
    out_include_dir = "include/opencv4",
    out_shared_libs = [
        "libopencv_calib3d.so",
        "libopencv_core.so",
        "libopencv_flann.so",
        "libopencv_features2d.so",
        "libopencv_highgui.so",
        "libopencv_imgcodecs.so",
        "libopencv_imgproc.so",
        "libopencv_video.so",
        "libopencv_videoio.so",
    ],
    visibility = ["//visibility:public"],
)
```

With `opencv_contrib` this is a bit more complicated. 

One approach is to build [shared libraries](opencv_build_script.sh) and 
then [link](link_opencv.sh) to local directory `third_party`.

