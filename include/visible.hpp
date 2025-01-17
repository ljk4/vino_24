#pragma once
#include "inference.hpp"

void show_points_result(cv::Mat& img,std::vector<Armor> armors_data );
void show_box_result(cv::Mat& img,std::vector<Armor> armors_data );
void show_number_result(cv::Mat& img,std::vector<Armor> armors_data);
