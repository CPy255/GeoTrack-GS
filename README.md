# GeoTrack-GS

本项目利用 3D 高斯溅射（3D Gaussian Splatting）技术，实现先进的 3D 场景重建与渲染。该项目基于论文《3D Gaussian Splatting for Real-Time Radiance Field Rendering》。此实现支持场景的训练和渲染，并利用 PyTorch 和自定义 CUDA 核心以实现高性能。

## 安装

1.  **克隆仓库：**

    ```bash
    git clone --recursive https://github.com/your-username/GeoTrack-GS.git
    cd GeoTrack-GS
    ```

2.  **创建并激活 Conda 环境：**

    ```bash
    conda env create -f environment.yml
    conda activate geotrack
    ```

3.  **构建自定义 CUDA 子模块：**

    本项目需要自定义的 CUDA 核心。请运行以下命令来构建和安装它们：

    ```bash
    cd submodules/diff-gaussian-rasterization-confidence
    python setup.py install
    cd ../../submodules/simple-knn
    python setup.py install
    cd ../.. 
    ```

## 使用方法

### 训练

要训练一个新模型，您需要一个已经由 COLMAP 处理过的数据集。

1.  **准备数据：**

    将您的数据集（例如，来自 COLMAP 的）放置在一个目录中。该目录结构通常应包含一个 `images` 文件夹和一个带有 COLMAP 重建结果的 `sparse` 文件夹。

2.  **运行训练脚本：**

    ```bash
    python train.py -s /path/to/your/dataset -m /path/to/output/model
    ```

    *   `-s, --source_path`: 输入数据集所在的目录路径。
    *   `-m, --model_path`: 用于保存训练好的模型和检查点的目录路径。

    更多训练选项，请参阅 `arguments/__init__.py` 中定义的参数。

### 渲染

当您拥有一个训练好的模型后，就可以从新的摄像机视角渲染图像。

```bash
python render.py -m /path/to/your/trained/model
```

### 评估

要评估一个训练好的模型，请使用 `full_eval.py` 脚本：

```bash
python full_eval.py -m /path/to/your/trained/model
```

## 许可证

本项目根据 LICENSE.md 文件中的条款进行许可。更多详情请参阅该文件。 
