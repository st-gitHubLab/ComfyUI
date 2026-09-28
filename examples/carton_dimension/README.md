# 纸箱视频尺寸回归工程

该示例实现流程图中的完整闭环：YOLO 每帧检测纸箱；仅当 BBox 完全进入画面时开始记录归一化的宽、高、面积；当检测框触及边界或消失时停止采集，聚合有效帧（均值、标准差、最大值，共 9 个特征），再由 MLP 输出 `Length, Width, Height`。

## 视频数据集

训练输入就是**带尺寸标签的视频**，而不是预先计算的特征。创建 `carton_dataset.csv`，每行一个视频及其真实尺寸（单位统一为 mm 或 cm）：

```csv
video,length,width,height
videos/carton_001.mp4,400,300,250
videos/carton_002.mp4,500,350,280
```

视频路径相对于 CSV 文件所在目录。每段视频应该只包含一个待测纸箱，并覆盖纸箱完整进入画面、在画面中央移动、再部分离开画面的过程。`length,width,height` 是人工测量的真实标签。

## 安装、训练与预测

```bash
cd examples/carton_dimension
python -m pip install -r requirements.txt
# 训练时会读取 CSV 中的视频，YOLO 自动提取 9 维特征，再训练 MLP
python train_mlp.py carton_dataset.csv --yolo carton_yolo.pt --carton-class 0 --output carton_mlp.pt
# 视频预测和带标注结果视频
python run.py input.mp4 --yolo carton_yolo.pt --mlp carton_mlp.pt --carton-class 0 --output output/predicted.mp4
# 直接对聚合后的 9 维特征执行 MLP 预测（用于排查模型）
python predict.py --mlp carton_mlp.pt --features "0.35,0.22,0.077,0.01,0.01,0.004,0.37,0.24,0.089"
```

### 预测代码

`run.py` 是正常生产预测入口：它从视频自动产生 9 维特征，并调用 `TorchMLPRegressor.predict()` 输出 `Length`、`Width`、`Height`。纸箱离开画面（或视频结束）后，结果会打印在终端；传入 `--output` 会保存带检测框、采集状态和预测值的标注 MP4。

`predict.py` 是独立的 MLP 预测入口，供排查模型使用；输入既可为逗号分隔的 9 个数字，也可为 `.npy` 或带 `features` 键的 `.npz` 文件。

`border_margin` 可过滤贴近边缘但尚未出画的抖动；`min_samples` 防止短暂检测触发回归。训练和推理时应保持这两个参数、相机、镜头、安装高度及纸箱距离一致。
