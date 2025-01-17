cmake_minimum_required(VERSION 3.19)
project(INFERENCE)

set(CMAKE_CXX_STANDARD 20)

# 设置OpenVINO和OpenCV的路径
set(OpenVINO_DIR "/opt/intel/openvino_2024/runtime/cmake")
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake")

# 设置包含目录和链接目录
set(INC_DIR "/opt/intel/openvino_2024/runtime/include")
set(LINK_DIR "/opt/intel/openvino_2024/runtime/lib/intel64")

# 源文件路径
set(SRC_DIR "src/inference.cpp" "src/video_detect_test.cpp" "src/visible.cpp")

# 查找OpenCV库
find_package(OpenCV 4 REQUIRED)

# 查找OpenVINO库
find_package(OpenVINO REQUIRED)

# 包含头文件目录
include_directories(
    ${INC_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${InferenceEngine_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
    "include/"
    "/opt/intel/openvino_2024/runtime/include/ie/"
    "/opt/intel/openvino_2024/runtime/include/ngraph/"
    "/opt/intel/openvino_2024/runtime/include/openvino/"
    /usr/include/ie
)

# 设置链接目录
link_directories(${LINK_DIR})

# 添加可执行文件
add_executable(INFERENCE ${SRC_DIR})

# 链接库
target_link_libraries(
    INFERENCE
    ${OpenCV_LIBS}
    openvino::runtime
)