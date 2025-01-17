#include "inference.hpp"
#include "visible.hpp"
int main(int argc, char* argv[])
{
    cv::VideoCapture capture(video_path);

    if (!capture.isOpened())
    {
        std::cout << "无法读取视频：" << argv[1] << std::endl;
        return -1;
    }
    // 读取视频帧，使用Mat类型的frame存储返回的帧
    cv::Mat frame;
    ArmorDetector armor_detector;

    while (true){

        capture >> frame;
        if (frame.empty())
        {
            std::cout << "视频读取完毕" << std::endl;
            break;
        }

        const int64 start = cv::getTickCount();
        std::cout<<"start infer"<<std::endl;
        armor_detector.startInferAndNMS(frame);
        std::cout<<"end infer"<<std::endl;
        // 计算FPS
        const float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms;" << std::endl;
        cv::putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);

        show_box_result(frame, armor_detector.get_armor());
        show_points_result(frame, armor_detector.get_armor());
        
        armor_detector.clear_armor();
        // 按下ESC键退出
        int k = cv::waitKey(10);
        if (k == 27)
        {
            std:: cout << "退出" << std::endl;
            break;
        }
        cv::imshow("result", frame);
    }
    return 0;
}
