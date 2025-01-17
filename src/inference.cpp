#include "inference.hpp"

//确定量，根据参数确定
static constexpr float scale = 1.0/(std::min( MODUL_INPUT_H*1.0/ INPUT_H,  MODUL_INPUT_W*1.0 / INPUT_W));
static const cv::Size new_unpad = cv::Size(int(round(INPUT_W / scale)), int(round(INPUT_H / scale)));
static const int dw = (MODUL_INPUT_W - new_unpad.width);
static const int dh = (MODUL_INPUT_H - new_unpad.height);

//构造函数，同时进行模型加载和预处理配置
ArmorDetector::ArmorDetector(){
    std::cout<<"initialize"<<std::endl;
    // -------- 加载模型 --------
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);
    // -------- 配置模型 --------
    ov::preprocess::PrePostProcessor ppp(model);
    //输入
    ppp.input().tensor()
    .set_element_type(input_type)
    .set_layout(input_layout)
    .set_color_format(input_ColorFormat)
    .set_shape(input_shape)
    ;
    //预处理依次按顺序执行，缩放必须放最前面，否则会增加耗时，pad必须在转换类型之前，否则会出现错误
    ppp.input().preprocess()
    .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, new_unpad.height, new_unpad.width)
    .convert_element_type(Moudel_type)
    .convert_layout(Moudel_layout)
    .pad({0, 0, 0, 0},          // batch, channel, height, width
         {0, 0, dh, dw}, 114.0f, ov::preprocess::PaddingMode::CONSTANT)
    .convert_color(Moudel_ColorFormat)
    .scale(255.0f)
    ;
    // //输出
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
}
//排序关键点
static void sort_keypoints(cv::Point2f keypoints[4]) {
    // Sort points based on their y-coordinates (ascending)
    std::sort(keypoints, keypoints + 4, [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });

    // Top points will be the first two, bottom points will be the last two
    cv::Point top_points[2] = { keypoints[0], keypoints[1] };
    cv::Point bottom_points[2] = { keypoints[2], keypoints[3] };

    // Sort the top points by their x-coordinates to distinguish left and right
    std::sort(top_points, top_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Sort the bottom points by their x-coordinates to distinguish left and right
    std::sort(bottom_points, bottom_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Assign sorted points back to the keypoints array
    keypoints[0] = top_points[0];     // top-left
    keypoints[1] = bottom_points[0];  // bottom-left
    keypoints[2] = bottom_points[1];  // bottom-right
    keypoints[3] = top_points[1];     // top-right
}

//进行推理
void ArmorDetector::startInferAndNMS(cv::Mat& img){
    // Set input tensor for model with one input
    ov::Tensor input_tensor(input_type, input_shape, img.ptr(0));

    infer_request.set_input_tensor(input_tensor);
    // -------- Start inference --------
    infer_request.infer();
    // -------- Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();

    // -------- Postprocess the result --------
    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,14]
    
    for (int cls=4 ; cls < (4+class_num); ++cls) {
        Armors armors;
        for (int i = 0; i < output_buffer.rows; i++) {
            //找出每个框的最大类别得分，共8400个框
            float class_score = output_buffer.at<float>(i, cls);
            float max_class_score = 0.0;
            for (int j = 4; j <  (4+class_num); j++) {
                if(max_class_score < output_buffer.at<float>(i, j)){
                    max_class_score = output_buffer.at<float>(i, j);
                }
            }
            if (class_score != max_class_score){
                continue;
            }

            if (class_score > BBOX_CONF_THRESH) {
                armors.class_scores_buffer.push_back(class_score);

                float cx = output_buffer.at<float>(i, 0);
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2);
                float h = output_buffer.at<float>(i, 3);

                // Get the box
                int left = int((cx - 0.5 * w-2) * scale);
                int top = int((cy - 0.5 * h - 2) * scale);
                int width = int(w * 1.2 * scale);
                int height = int(h * 1.2 * scale);

                // Get the keypoints
                std::vector<float> keypoints;
                cv::Mat kpts = output_buffer.row(i).colRange( (4+class_num), output_buffer.cols);
                for (int i = 0; i < 4; i++) {
                    float x = kpts.at<float>(0, i * 2 + 0) * scale;
                    float y = kpts.at<float>(0, i * 2 + 1) * scale;

                    keypoints.push_back(x);
                    keypoints.push_back(y);
                }
                armors.boxes_buffer.push_back(cv::Rect(left, top, width, height));
                armors.objects_keypoints_buffer.push_back(keypoints);
            }
        }
        armors.class_ids = cls - 4;
        //NMS处理
        std::vector<int> indices;
        cv::dnn::NMSBoxes(armors.boxes_buffer, armors.class_scores_buffer, BBOX_CONF_THRESH, NMS_THRESH, indices);

        for(auto i:indices)
        {
            Armor armor;

            armor.box= armors.boxes_buffer[i];
            armor.class_scores = armors.class_scores_buffer[i];
            armor.class_ids = armors.class_ids;

            for (int j = 0; j < 4; j++) {
                int x = std::clamp(int(armors.objects_keypoints_buffer[i][j * 2 + 0]), 0, INPUT_W);
                int y = std::clamp(int(armors.objects_keypoints_buffer[i][j * 2 + 1]), 0, INPUT_H);
                armor.objects_keypoints[j] = cv::Point(x, y);
            }
            sort_keypoints(armor.objects_keypoints);
            last_Armors.push_back(armor);
        }
    }
}

std::vector<Armor> ArmorDetector::get_armor()
{
    return last_Armors;
}

void ArmorDetector::clear_armor()
{
    last_Armors.clear();
}


