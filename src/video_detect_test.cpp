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
    int fps = 0;
    while (true){

        capture >> frame;
        if (frame.empty())
        {
            std::cout << "视频读取完毕" << std::endl;
            break;
        }
        
        cv::TickMeter tm;
        tm.start();

        std::cout<<"start infer"<<std::endl;
        armor_detector.startInferAndNMS(frame);

        tm.stop();
        std::cout << "time cost: " << tm.getTimeMilli() << "ms" << std::endl;
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
