# Wan Long Video Generator

一个用于长视频生成的工程骨架：第一段使用 Wan2.1 T2V 文生视频，后续每段提取上一段最后一帧，再使用 Wan2.1 I2V 图生视频续接，最后用 FFmpeg 拼接为完整视频。

## 目录结构

```text
Wan_Long_Video_Generator/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── core/
│   ├── pipeline.py
│   ├── wan_t2v.py
│   ├── wan_i2v.py
│   ├── storyboard.py
│   ├── frame_extract.py
│   ├── ffmpeg.py
│   └── utils.py
├── outputs/
│   ├── clips/
│   ├── frames/
│   └── final/
└── models/
    ├── Wan2.1-T2V-1.3B/
    └── Wan2.1-I2V-1.3B/
```

## 快速开始

1. 安装依赖：

   ```bash
   cd Wan_Long_Video_Generator
   pip install -r requirements.txt
   ```

2. 安装 FFmpeg，并确保 `ffmpeg` 命令可用。
3. 将 Wan2.1 模型文件分别放入：
   - `models/Wan2.1-T2V-1.3B/`
   - `models/Wan2.1-I2V-1.3B/`
4. 在 `core/wan_t2v.py` 和 `core/wan_i2v.py` 中接入你的本地 Wan2.1 推理代码。
5. 启动界面：

   ```bash
   python app.py
   ```

## 流程说明

- `core/storyboard.py`：按 5 秒一段自动生成分镜提示词。
- `core/wan_t2v.py`：第一段视频的文生视频入口。
- `core/frame_extract.py`：提取每段最后一帧作为续接关键帧。
- `core/wan_i2v.py`：后续片段的图生视频入口。
- `core/ffmpeg.py`：拼接所有片段生成最终 MP4。
- `core/pipeline.py`：串联上述步骤的总流程。

> 注意：当前提交提供可运行的工程骨架和清晰扩展点，Wan2.1 的实际推理调用需要根据你本地模型实现补充。
