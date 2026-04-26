# Project Log

2026-04-26: 对接了队友更新后的模型接口，调整 src/data/dataset.py 和 src/train.py，使训练入口优先读取处理后的真实数据并保持 SampleBatch(inputs, target) 一致。
同时整理了数据处理链路，保留 preprocess 与 visualize 的最新版本，并删除了临时的 dataset 备份文件以统一仓库状态。