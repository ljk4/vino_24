#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <memory>
#include <openvino/openvino.hpp>

using namespace std;
using namespace cv;

static  string model_path = "/home/ljk/rm_model/v8_armor/best.onnx";
static  string video_path = "/home/ljk/armr_test/video_test/test.mp4";
static constexpr int INPUT_W = 1280;    // 输入图像的宽度
static constexpr int INPUT_H = 1024;    // 输入图像的高度
static constexpr int MODUL_INPUT_W = 416;    // 输入图像的宽度
static constexpr int MODUL_INPUT_H = 416;    // 输入图像的高度
static constexpr int NUM_CLASSES = 2;  // 分类的数量

static constexpr float BBOX_CONF_THRESH = 0.7;  // 边界框的置信度阈值
static constexpr float NMS_THRESH = 0.3;  // 非极大值抑制（NMS）的阈值

static constexpr float MERGE_CONF_ERROR = 0.15;  // 合并过程中允许的置信度误差
static constexpr float MERGE_MIN_IOU = 0.8;  // 合并过程中最小的交并比（IoU）

static constexpr int TOPK = 4;  // 保留的最高置信度的边界框数量


static ov::element::Type input_type = ov::element::u8;
static ov::Shape input_shape = {1, MODUL_INPUT_H, MODUL_INPUT_W, 3};
static const ov::Layout input_layout{"NHWC"};
static ov::preprocess::ColorFormat input_ColorFormat = ov::preprocess::ColorFormat::BGR;

static ov::element::Type Moudel_type = ov::element::f32;
static ov::Shape Moudel_shape = {1, 3, MODUL_INPUT_H, MODUL_INPUT_W};
static const ov::Layout Moudel_layout{"NCHW"};
static ov::preprocess::ColorFormat Moudel_ColorFormat = ov::preprocess::ColorFormat::RGB;

struct Armor{
    float class_scores;
    cv::Rect box;
    cv::Point2f objects_keypoints[4];
    int class_ids;//color
    cv::Mat number_img;
    std::string number;

};

struct Armors
{
    std::vector<float> class_scores_buffer;
    std::vector<cv::Rect> boxes_buffer;
    std::vector<std::vector<float>> objects_keypoints_buffer;
    int class_ids;
};

class ArmorDetector
{
public:
    ArmorDetector();
    ~ArmorDetector();
    void preprocess_img(Mat& inframe);
    void startInferAndNMS();
    std::vector<Armor> get_armor();
    void clear_armor();
private:
    cv::Mat blob;
    ov::InferRequest infer_request;
    std::vector<Armor> last_Armors;
    ov::Output<const ov::Node> input_port;
    cv::Size shape;
    float scale = 1.0/(std::min( MODUL_INPUT_H*1.0/ INPUT_H,  MODUL_INPUT_W*1.0 / INPUT_W));
    cv::Size new_unpad = cv::Size(int(round(INPUT_W / scale)), int(round(INPUT_H / scale)));
    int dw = (MODUL_INPUT_W - new_unpad.width) ;
    int dh = (MODUL_INPUT_H - new_unpad.height) ;

};