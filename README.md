# 装甲版推理代码
openvino2024版

## 使用方法

1. 确定模型参数，使用以下命令，将 `last.xml` 替换为自己的模型：
    ```sh
    benchmark_app -m last.xml -d CPU -niter 1
    ```

2. 确定输入图像的参数，换视频时记得改尺寸。

3. 更改模型和视频路径。

4. 在 `src/inference.cpp` 中修改参数，检测参数设置是否正确。

5. 编译运行：
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ./INFERENCE
    ```
## 相关函数

模型初始化和配置在构造函数中完成。

- `armor_detector.startInferAndNMS(frame);`  
    运行此函数进行推理并将结果保存到 `last_Armors` 中。

- `std::vector<Armor> armors_data = armor_detector.get_armor();`  
    运行此函数获取检测结果。`Armors` 表示一个框，其定义如下：
    
    ```cpp
    struct Armors {
            std::vector<float> class_scores_buffer;
            std::vector<cv::Rect> boxes_buffer;
            std::vector<std::vector<float>> objects_keypoints_buffer;
            int class_ids; //0:blue 1:red
    };
    ```

- `show_box_result(frame, armor_detector.get_armor());`  
    用于可视化检测框。

- `show_points_result(frame, armor_detector.get_armor());`  
    用于可视化灯条点。

- `armor_detector.clear_armor();`  
    此函数用于清空本帧检测结果。

