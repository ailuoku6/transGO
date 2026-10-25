# 接续

从 index.md 和 README 的“Mac NPU / Neural Engine 推理”开始。

- 可选Core ML CPU_AND_NE后端已接入所有推理入口；训练保持PyTorch/MPS、默认torch、旧schema兼容。
- 核心文件coreml.py、search.py、cli.py、pipeline.py、actors.py、evaluate.py；tests/test_coreml.py覆盖验证。
- 本机现有9x9权重局部搜索中位约3.1倍加速；benchmark.json保留原始结果。integration.json为真实并行自博弈/MPS训练/Core ML评估一轮成功记录。
- .venv已安装coremltools9.0，PyTorch2.14.1超出其官方已测范围但固定图已真实验证。沙箱可能不可访问原生编译缓存，硬件测试须可访问macOS服务。
- 本轮完成：标准198 passed/8 skipped，3/9/19路真实硬件3 passed；完整双worker闭环通过。后续只有可选长时间吞吐/功耗/棋力实验。
- 不把CPU_AND_NE当作NPU独占、不把静态225算子当作实时利用率；不把局部基准当作整轮/功耗/棋力结论。
- 不修改device为npu，不迁移训练到Core ML，不替用户重启长训练。新命令显式加--inference-backend coreml。
