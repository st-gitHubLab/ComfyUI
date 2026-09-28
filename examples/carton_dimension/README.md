# 纸箱视频尺寸回归工程

该示例实现流程图中的完整闭环：YOLO 每帧检测纸箱；仅当 BBox 完全进入画面时开始记录归一化的宽、高、面积；当检测框触及边界或消失时停止采集，聚合有效帧（均值、标准差、最大值，共 9 个特征），再由 MLP 输出 `Length, Width, Height`。

## 安装、训练与预测

```bash
cd examples/carton_dimension
python -m pip install -r requirements.txt
python train_mlp.py carton_samples.npz --output carton_mlp.pt
python run.py input.mp4 --yolo carton_yolo.pt --mlp carton_mlp.pt --carton-class 0 --output output/predicted.mp4
# 直接对聚合后的 9 维特征执行 MLP 预测
python predict.py --mlp carton_mlp.pt --features "0.35,0.22,0.077,0.01,0.01,0.004,0.37,0.24,0.089"
```

训练数据 `carton_samples.npz` 必须含有：`features`（`N×9`，按运行时同样的聚合规则产生）和 `targets`（`N×3`，依次为长、宽、高，单位自定但通常为 mm）。YOLO 权重必须含纸箱类别。为达到实际计量精度，应使用同一相机、镜头、安装高度和纸箱距离采集覆盖目标尺寸范围的标注数据。

### 预测代码

`predict.py` 是独立的 MLP 预测入口：加载 `carton_mlp.pt`，归一化 9 维输入，调用 `TorchMLPRegressor.predict()`，并输出 JSON：`{"length": ..., "width": ..., "height": ...}`。输入既可为逗号分隔的 9 个数字，也可为 `.npy` 或带 `features` 键的 `.npz` 文件。`run.py` 则负责从视频中自动产生同一组 9 维特征后调用相同的预测代码。

运行时会在纸箱离开画面（或视频结束时）直接在终端输出预测的 `Length`、`Width`、`Height` 与有效帧数；传入 `--output` 会保存带有检测框、采集状态和预测值的标注 MP4。

`border_margin` 可过滤贴近边缘但尚未出画的抖动；`min_samples` 防止短暂检测触发回归。它们可在 `CartonDimensionPipeline` 中按现场视频帧率和目标速度调整。
