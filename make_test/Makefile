# Smoke test to build and run Aruco
OPENCV_DIR = install
OPENCV_INCLUDE = $(OPENCV_DIR)/include/opencv4
OPENCV_LIB = $(OPENCV_DIR)/lib

# Compiler settings
# This is important -- lopencv_aruco MUST BE FIRST
CXX = g++
CXXFLAGS = -std=c++11 -I$(OPENCV_INCLUDE)
LDFLAGS = -L$(OPENCV_LIB) -Wl,-rpath,$(OPENCV_LIB)
LIBS = -lopencv_aruco -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core


# Target
TARGET = test_aruco
SOURCE = test_aruco.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	./$(TARGET)

.PHONY: clean test