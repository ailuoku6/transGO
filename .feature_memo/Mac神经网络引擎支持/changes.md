# 变更与验证

## 2026-10-25

- 新增可选 Core ML 后端、所有推理入口、内容缓存、批处理、CPU合法mask和静态设备计划，Config/checkpoint/replay schema 不变。
- .venv安装coremltools9.0；修复PyTorch2.14/NumPy2.5动态shape int转换问题。未降级原PyTorch。
- 当前标准套件198 passed、8 skipped；跳过5个沙箱中无MPS的用例和3个显式开启的Core ML硬件用例。
- 3/9/19路真实转换、全部预测头/归一化概率和缓存复用最终3 passed（TRANSFORMER_GO_TEST_COREML=1）。FP16 raw logits观察到约0.0113偏差；raw与概率分别约束，概率绝对误差<=0.005，搜索接口沿用更严格容差。
- step327000、9x9空盘200次搜索：MPS三次0.3568/0.3067/0.3204秒；Core ML batch16为0.1137/0.1038/0.1036秒，约3.086倍中位加速。batch1为0.2049/0.2086/0.1477秒。225个算子静态首选NE。原始benchmark.json。
- 最终CLI benchmark --coreml-compute-plan实跑成功，报告coreml-cpu-and-ne、225个NE偏好算子；该独立单次约0.152秒，运行负载有波动。
- 完整原生并行闭环：/private/tmp/transformergo-coreml-loop.GIRCIm，2 worker生成2局replay、MPS训练1步、2 worker Core ML评估2局，iteration成功。integration.json保存摘要。
- 没有测实时NPU占用率、功耗、长时间吞吐或棋力；原有runs/zero-9x9仅被读取，未改写/重启已有训练。
