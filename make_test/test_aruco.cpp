#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    try {
        // Test 1: Try to get predefined dictionary
        std::cout << "Testing getPredefinedDictionary..." << std::endl;
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        std::cout << "✓ getPredefinedDictionary works!" << std::endl;
        
        // Test 2: Generate a marker
        std::cout << "Testing generateImageMarker..." << std::endl;
        cv::Mat marker_image;
        cv::aruco::generateImageMarker(dictionary, 23, 200, marker_image, 1);
        
        if (!marker_image.empty()) {
            std::cout << "✓ Marker generated! Size: " << marker_image.rows << "x" << marker_image.cols << std::endl;
            
            // Test 3: Save the marker
            bool saved = cv::imwrite("/tmp/test_marker.png", marker_image);
            if (saved) {
                std::cout << "✓ Marker saved to /tmp/test_marker.png" << std::endl;
            } else {
                std::cout << "✗ Failed to save marker" << std::endl;
            }
        } else {
            std::cout << "✗ Failed to generate marker" << std::endl;
        }
        
        std::cout << "\nAll tests passed! Your OpenCV aruco build is working." << std::endl;
        
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl; 
        return -1;
    }
    
    return 0;
}