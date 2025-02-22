#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <memory>
#include <openvino/openvino.hpp>
#include <openvino/op/non_max_suppression.hpp>
#include <openvino/op/ops.hpp>
#include <openvino/runtime/runtime.hpp>

static const std::string model_path = "/home/ljk/rm_model/new_25_mode/best.onnx";
static const std::string video_path = "/home/ljk/rm_model/6.mp4";
static constexpr int MODUL_INPUT_W = 640;    // 模型图像的宽度
static constexpr int MODUL_INPUT_H = 640;    // 模型图像的高度

static constexpr float BBOX_CONF_THRESH = 0.7;  // 边界框的置信度阈值
static constexpr float NMS_THRESH = 0.3;  // 非极大值抑制（NMS）的阈值

static constexpr int class_num = 2; // 类别数
static const std::vector<std::string> class_names = {"blue","red"}; // 类别名称 

static const ov::element::Type input_type = ov::element::u8;
static const ov::Layout input_layout{"NHWC"};
static const ov::preprocess::ColorFormat input_ColorFormat = ov::preprocess::ColorFormat::BGR;

static const ov::element::Type Moudel_type = ov::element::f32;
static const ov::Shape Moudel_shape = {1, 3, MODUL_INPUT_H, MODUL_INPUT_W};
static const ov::Layout Moudel_layout{"NCHW"};
static const ov::preprocess::ColorFormat Moudel_ColorFormat = ov::preprocess::ColorFormat::RGB;

struct Armor{
    float class_scores;
    cv::Rect box;
    cv::Point2f objects_keypoints[4];
    int class_ids;//color
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
    void init(const size_t &input_w,const size_t &input_h);
    void startInferAndNMS(cv::Mat& inframe);
    std::vector<Armor> get_armor();
    void clear_armor();
    void initNMSModel();
private:
    ov::InferRequest infer_request;
    ov::CompiledModel nms_compiled_model;
    ov::InferRequest nms_infer_request;
    std::vector<Armor> last_Armors;
    size_t INPUT_W;
    size_t INPUT_H;
    ov::Shape input_shape;
    float scale;
};

