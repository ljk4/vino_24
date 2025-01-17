#include "visible.hpp"
#include "inference.hpp"

void show_points_result(cv::Mat& img,std::vector<Armor> armors_data ) {
    for (auto i: armors_data) {
       for(int j=0;j<4;j++){
           cv::line(img, i.objects_keypoints[j], i.objects_keypoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
       }
       for(int j=0;j<4;j++){
           if(j == 0){
               cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 0, 255), -1);
           }else if(j==1){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 0, 0), -1);}
           else if(j==2){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(0, 255, 0), -1);}
           else if(j==3){     cv::circle(img, i.objects_keypoints[j], 2, cv::Scalar(255, 0, 0), -1);}
       }
       cv::putText(img, class_names.at(i.class_ids), i.objects_keypoints[0], cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
}
void show_box_result(cv::Mat& img,std::vector<Armor> armors_data ) {
    for(auto i: armors_data){
        cv::rectangle(img, i.box, cv::Scalar(0, 255, 0), 2);
    }
}

