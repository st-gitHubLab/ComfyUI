# 纸箱视频尺寸回归工程

该示例实现流程图中的完整闭环：YOLO 每帧检测纸箱；仅当 BBox 完全进入画面时记录归一化的宽、高、面积；检测框触及边界或消失时停止采集，聚合有效帧（均值、标准差、最大值，共 9 个特征），再由 MLP 输出 `Length, Width, Height`。

## 只需修改一个配置文件

**不需要在命令行输入模型、视频或数据集参数。** 所有路径和参数都在 [`config.py`](config.py) 中：

- `YOLO_WEIGHTS`：纸箱 YOLO 模型；
- `MLP_CHECKPOINT`：训练生成的尺寸回归模型；
- `DATASET_CSV`：训练视频数据集清单；
- `INPUT_VIDEO` 与 `OUTPUT_VIDEO`：预测输入和标注结果视频；
- `CARTON_CLASS`、`BORDER_MARGIN`、`MIN_SAMPLES`：YOLO 类别和特征采集参数；
- `PREDICTION_FEATURES`：仅供 `predict.py` 单独验证 MLP 的 9 维输入。

默认约定是把 YOLO 权重放在 `models/carton_yolo.pt`，训练后 MLP 写入 `models/carton_mlp.pt`，视频放在 `data/` 下。请先按现场路径和类别编号编辑 `config.py`。

## 视频数据集

训练输入是**带尺寸标签的视频**。在 `config.py` 所指向的位置创建 CSV，每行一个视频及真实尺寸（单位统一为 mm 或 cm）：

```csv
video,length,width,height
videos/carton_001.mp4,400,300,250
videos/carton_002.mp4,500,350,280
```

视频路径相对于 CSV 文件所在目录。每段视频应该只包含一个待测纸箱，并覆盖纸箱完整进入画面、在画面中央移动、再部分离开画面的过程。`length,width,height` 是人工测量的真实标签。

## 运行

```bash
cd examples/carton_dimension
python -m pip install -r requirements.txt
python train_mlp.py  # 从 config.py 的 DATASET_CSV 视频训练，保存到 MLP_CHECKPOINT
python run.py        # 对 config.py 的 INPUT_VIDEO 预测，输出 OUTPUT_VIDEO
python predict.py    # 使用 config.py 的 PREDICTION_FEATURES 单独验证 MLP
```

`run.py` 会在纸箱离开画面（或视频结束）后打印 `Length`、`Width`、`Height` 和有效帧数，并保存含检测框、采集状态和预测值的标注 MP4。训练和推理时应保持 `BORDER_MARGIN`、`MIN_SAMPLES`、相机、镜头、安装高度及纸箱距离一致。
