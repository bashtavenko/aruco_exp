#!/bin/bash

set -e

# Parse command line argument
BUILD_TYPE="Release"
if [ "$1" = "debug" ]; then
    BUILD_TYPE="Debug"
    echo "Building in Debug mode..."
elif [ "$1" != "" ] && [ "$1" != "release" ]; then
    echo "Usage: $0 [debug|release]"
    echo "  debug   - Build debug version with symbols"
    echo "  release - Build optimized release version (default)"
    exit 1
fi

# Configuration
OPENCV_VERSION="4.8.0"
BUILD_DIR="$HOME/opencv_build"
if [ "$BUILD_TYPE" = "Debug" ]; then
    INSTALL_DIR="$BUILD_DIR/install_debug"
    BUILD_SUBDIR="build_debug"
else
    INSTALL_DIR="$BUILD_DIR/install"
    BUILD_SUBDIR="build"
fi

echo "Building OpenCV ${OPENCV_VERSION} with ArUco module in $BUILD_TYPE mode..."

# Create and enter build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Download sources if they don't exist
if [ ! -d "opencv-${OPENCV_VERSION}" ]; then
    echo "Downloading OpenCV ${OPENCV_VERSION}..."
    curl -L "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz" -o "opencv-${OPENCV_VERSION}.tar.gz"
    tar -xzf "opencv-${OPENCV_VERSION}.tar.gz"
fi

if [ ! -d "opencv_contrib-${OPENCV_VERSION}" ]; then
    echo "Downloading OpenCV Contrib ${OPENCV_VERSION}..."
    curl -L "https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.tar.gz" -o "opencv_contrib-${OPENCV_VERSION}.tar.gz"
    tar -xzf "opencv_contrib-${OPENCV_VERSION}.tar.gz"
fi

# Create build directory
mkdir -p "$BUILD_SUBDIR"
cd "$BUILD_SUBDIR"

echo "Configuring CMake for $BUILD_TYPE build..."
cmake \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DBUILD_LIST=calib3d,core,features2d,highgui,imgcodecs,imgproc,video,videoio,aruco,objdetect \
    -DOPENCV_EXTRA_MODULES_PATH="../opencv_contrib-${OPENCV_VERSION}/modules" \
    -DWITH_FFMPEG=ON \
    -DWITH_GTK=OFF \
    -DWITH_QT=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    "../opencv-${OPENCV_VERSION}"

echo "Building OpenCV..."
make -j$(nproc)

echo "Installing OpenCV..."
make install

echo "Creating .408 version symlinks..."
cd "$INSTALL_DIR/lib"
for lib in libopencv_*.so.4.8.0; do
    if [ -f "$lib" ]; then
        base=$(basename "$lib" .so.4.8.0)
        ln -sf "$lib" "${base}.so.408"
        echo "Created symlink: ${base}.so.408 -> $lib"
    fi
done

echo ""
echo "Build completed successfully!"
echo "Build type: $BUILD_TYPE"
echo "Libraries installed in: $INSTALL_DIR/lib"
echo "Headers installed in: $INSTALL_DIR/include"
echo ""
echo "Required libraries:"
ls -la "$INSTALL_DIR/lib"/libopencv_*.so.408 2>/dev/null || echo "No .408 symlinks found"

if [ "$BUILD_TYPE" = "Debug" ]; then
    echo ""
    echo "Debug build complete. Use this path for debugging:"
    echo "  Libraries: $INSTALL_DIR/lib"
    echo "  Headers: $INSTALL_DIR/include"
fi