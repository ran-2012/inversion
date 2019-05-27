## 一维航空瞬变电磁数据反演：使用CUDA加速

### 本项目旨在最大化硬件利用率来计算复杂的地球物理反演问题

#### 项目整体说明

  - CUDA计算正演
  - pybind11结合C++与Python
  - TensorFlow框架计算反演

#### 模块说明
    
  - forward_gpu: CUDA计算正演部分
  - forward: 连接CUDA与C++，以实现两个编译器分离编译
  - data: 正反演过程中的数据抽象
  - test: 测试（尚不完备）
  - global: 通用辅助函数
  - inversion: Python反演模块（尚不完备）

#### 使用说明

  - 在C++层的使用方法可参考 /test/test.cpp 中的的代码
  - 在Python层的使用方法可参考 /inversion/test.py 中的代码