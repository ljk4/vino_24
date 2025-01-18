# 装甲版推理代码

openvino2024版

在测试视频上的耗时：
- Total frames: 881
- Max infer time: 12.0828ms
- Min infer time: 5.64025ms
- Average infer time: 6.84996ms

## 使用方法

1. 确定模型参数，使用以下命令，将 `last.xml` 替换为自己的模型：
    ```sh
    benchmark_app -m last.xml -d CPU -niter 1
    ```

2. 更改模型和视频路径。

3. 在 `src/inference.cpp` 中修改参数，检测参数设置是否正确。

4. 编译运行：
    ```sh
    mkdir build
    cd build
    cmake ..
    make
    ./INFERENCE
    ```

## 相关函数

- `armor_detector.init(frame.cols, frame.rows);`  
  模型初始化和配置在此函数中完成，通过静态局部变量保证只运行一次。

- `armor_detector.startInferAndNMS(frame);`  
  运行此函数进行推理并将结果保存到 `last_Armors` 中。

- `std::vector<Armor> armors_data = armor_detector.get_armor();`  
  运行此函数获取检测结果。`Armors` 表示一个框，其定义如下：
    
    ```cpp
    struct Armors {
        std::vector<float> class_scores_buffer;
        std::vector<cv::Rect> boxes_buffer;
        std::vector<std::vector<float>> objects_keypoints_buffer;
        int class_ids; // 0: blue, 1: red
    };
    ```

- `show_box_result(frame, armor_detector.get_armor());`  
  用于可视化检测框。

- `show_points_result(frame, armor_detector.get_armor());`  
  用于可视化灯条点。

- `armor_detector.clear_armor();`  
  此函数用于清空本帧检测结果。

