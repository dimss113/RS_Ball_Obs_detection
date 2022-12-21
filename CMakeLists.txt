cmake_minimum_required(VERSION 3.14)
project("vision_realsense" CXX)

find_package(OpenCV REQUIRED)
find_package(NumCpp 2.8.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(DEPENDENCIES realsense2 ${OpenCV_LIBS})

message( STATUS "OPENCV VERSION ${OpenCV_VERISON}")

add_executable(Vision_realsense src/vision_realsense2.cpp)
# add_executable(Vision_realsense2 vision_realsense.cpp)
# add_executable(frame_test frameTest.cpp)
set_property(TARGET Vision_realsense PROPERTY CXX_STANDARD 11)
# set_property(TARGET Vision_realsense2 PROPERTY CXX_STANDARD 11)
target_link_libraries(Vision_realsense 
    ${DEPENDENCIES}
    ${OpenCV_LIBS}
    ${PYTHON_LIBRARIES}
    NumCpp::NumCpp
    # Python3::NumPy
)
# target_link_libraries(Vision_realsense2
#     ${DEPENDENCIES}
#     ${OpenCV_LIBS}
#     ${PYTHON_LIBRARIES}
#     NumCpp::NumCpp
#     # Python3::NumPy
# )