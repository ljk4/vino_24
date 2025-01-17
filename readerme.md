# 装甲版推理代码
openvino2024版
## 使用方法
1. 确定模型参数，使用 ```benchmark_app -m last.xml -d CPU -niter 1  ```将last.xml替换为自己的模型
2. 确定输入图像的参数
3. 更改模型和视频路径
4. 在src/inference.cpp中修改参数，检测参数设置是否正确
5. 编译运行即可