"""《对照 transGO 学习机器学习》配套实验脚本。

每个实验对应电子书中的一课，只使用 CPU 和小型教学模型。
运行方式（在项目根目录执行）：
    .venv/bin/python scripts/ml_learning_lab.py --lesson linear
把 linear 换成其他实验名；all 表示依次运行全部实验。

安全说明：不读取、不修改正式训练的权重、replay 数据或配置。
只有 train 实验会写文件，而且写在自动删除的临时目录里。
"""
import argparse
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# 项目根目录。加入导入路径后，从任何目录运行脚本都能找到 transformer_go 包。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from transformer_go._native import Board
from transformer_go.checkpoint import load_checkpoint
from transformer_go.config import load_config
from transformer_go.data import augment, read_shard
from transformer_go.model import PolicyValueNet
from transformer_go.runtime import seed_all
from transformer_go.search import Evaluator
from transformer_go.selfplay import generate
from transformer_go.train import _ownership_targets, _smoothed_one_hot, losses, train


def smoke_model():
    """创建教学用小模型：3 路棋盘、约 4 万个参数、固定随机种子、只用 CPU。"""
    config = load_config(ROOT / "configs/smoke.yaml")
    config.device = "cpu"
    seed_all(config.seed)
    return config, PolicyValueNet(config.model)


def demo_linear():
    """第 01 课：输入不变、参数改变，预测就会改变。"""
    # 输入
    x = torch.tensor(2.0)
    # 参数 w
    weight = torch.tensor(3.0)
    # 参数 b
    bias = torch.tensor(1.0)
    # y = wx + b = 7
    prediction = weight * x + bias
    # 只把 w 改成 4，预测变成 9
    second_prediction = torch.tensor(4.0) * x + bias
    return {"input": x.item(), "weight": weight.item(), "bias": bias.item(),
            "prediction": prediction.item(), "second_weight": 4.0,
            "second_prediction": second_prediction.item()}


def demo_tensors():
    """第 02 课：真实棋盘怎样变成网络输入张量。"""
    # 3 路空棋盘，黑方下在动作 0（左上角）；下一手轮到白方。
    board = Board(3, 0.5).play(0)
    # 形状 [21,3,3]：21 张信息图
    features = board.features()
    # 前面加 batch 轴，变成 [1,21,3,3]
    batch = torch.from_numpy(features).unsqueeze(0)
    # 9 个棋盘点 + pass，共 10 个动作
    legal = torch.tensor(board.legal_mask(), dtype=torch.bool)
    return {"to_play": board.to_play, "cells": board.cells,
            "features_shape": list(features.shape), "batch_shape": list(batch.shape),
            "dtype": str(batch.dtype),
            # 轮到白方，所以左上角的黑子出现在“对方棋子”通道 1，而不是通道 0。
            "current_stone_at_0": float(features[0, 0, 0]),
            "opponent_stone_at_0": float(features[1, 0, 0]),
            # 通道 18 表示“当前是否黑方行棋”，此时全为 0。
            "black_to_play_plane": float(features[18, 0, 0]),
            # 左上角已有棋子，动作 0 不合法。
            "legal_shape": list(legal.shape), "action_0_legal": bool(legal[0])}


def demo_probabilities():
    """第 03 课：logits 经 softmax 变成概率，合法 mask 让非法动作概率为 0。"""
    # 网络原始分数，不是概率
    logits = torch.tensor([2.0, 1.0, 0.0])
    # 假设第二个动作非法
    legal = torch.tensor([True, False, True])
    # 非法动作分数设为负无穷
    masked = logits.masked_fill(~legal, float("-inf"))
    # exp(-inf)=0，非法动作概率严格为 0
    probability = masked.softmax(-1)
    # 项目顺序：负、和、胜
    wdl = torch.tensor([0.2, 0.1, 0.7])
    return {"logits": logits.tolist(), "probability": logits.softmax(-1).tolist(),
            "legal": legal.tolist(), "masked_probability": probability.tolist(),
            "masked_sum": probability.sum().item(), "wdl_loss_draw_win": wdl.tolist(),
            # 搜索使用的价值 = 胜概率 - 负概率，不是胜率。
            "value": (wdl[2] - wdl[0]).item(),
            # 和棋算半分时的预测得分率。
            "predicted_match_points": (wdl[2] + 0.5 * wdl[1]).item()}


def demo_gradient():
    """第 04 课：手算一次“预测、算损失、求梯度、更新参数”。"""
    # requires_grad=True：要求 PyTorch 记录 weight 参与的计算，以便求梯度。
    weight = torch.tensor(1.0, requires_grad=True)
    x, target = torch.tensor(2.0), torch.tensor(6.0)
    # 预测 2
    prediction = weight * x
    # 平方误差 (2-6)^2 = 16
    loss = (prediction - target).square()
    # 反向传播：算出 weight.grad = -16，但还不改 weight
    loss.backward()
    gradient = weight.grad.item()
    weight_before = weight.item()
    # 这里故意使用最简单的梯度下降，不是项目实际使用的 AdamW。
    # 新参数 = 旧参数 - 学习率 * 梯度 = 1 - 0.1 * (-16) = 2.6
    # 更新参数这一步本身不需要再记录梯度
    with torch.no_grad():
        weight -= 0.1 * weight.grad
        # 新预测 5.2
        after = weight * x
        # 新损失约 0.64
        after_loss = (after - target).square()
    return {"input": x.item(), "target": target.item(), "learning_rate": 0.1,
            "weight_before": weight_before, "prediction_before": prediction.item(),
            "loss_before": loss.item(), "gradient": gradient,
            "weight_after": weight.item(), "prediction_after": after.item(),
            "loss_after": after_loss.item()}


def demo_forward():
    """第 05～06 课：跟踪真实网络一次前向计算中各层的输出形状。"""
    _, model = smoke_model()
    # 推理模式：只预测，不训练
    model.eval()
    board = Board(3, 0.5).play(0)
    features = torch.from_numpy(board.features()).unsqueeze(0)
    legal = torch.tensor([board.legal_mask()], dtype=torch.bool)
    stages = {"input": list(features.shape)}

    def record(name):
        # forward hook：某一层算完时被调用，这里只记录输出形状，不改变计算结果。
        def hook(_module, _inputs, output):
            stages[name] = list(output.shape)
        return hook

    # 卷积入口：[1,32,3,3]
    handles = [model.stem.register_forward_hook(record("stem")),
               # 9 个点 + pass：[1,10,32]
               model.blocks[0].register_forward_hook(record("block_1")),
               model.norm.register_forward_hook(record("final_norm"))]
    try:
        # 不记录梯度，节省计算
        with torch.inference_mode():
            outputs = model(features, legal)
            probability = outputs["policy_logits"].softmax(-1)
    finally:
        # 用完移除 hook，避免影响之后的计算
        for handle in handles:
            handle.remove()
    return {"stages": stages, "output_shapes": {k: list(v.shape) for k, v in outputs.items()},
            "parameters": sum(p.numel() for p in model.parameters()),
            "policy_sum": probability.sum().item(), "illegal_probability": probability[0, 0].item(),
            "note": "未训练的教学模型：形状有参考价值，预测概率不代表棋力。"}


def demo_attention():
    """第 06 课：用两个位置、一维向量手算一次 attention。"""
    # Q：两个位置各自“拿什么去比较”
    query = torch.tensor([[1.0], [0.0]])
    # K：两个位置各自“提供什么供匹配”
    key = torch.tensor([[1.0], [0.0]])
    # V：匹配后实际汇总的信息
    value = torch.tensor([[10.0], [20.0]])
    # 点积得到匹配分数，再除以 sqrt(维度) 做缩放。
    scores = query @ key.T / math.sqrt(query.shape[-1])
    # 每个位置的参考权重，总和为 1
    weights = scores.softmax(-1)
    # 屏蔽第一个位置对第二个位置的参考，观察汇总结果如何变化。
    masked_scores = scores.clone()
    masked_scores[0, 1] = float("-inf")
    masked_weights = masked_scores.softmax(-1)
    return {"query": query.tolist(), "key": key.tolist(), "value_vectors": value.tolist(),
            "scores": scores.tolist(), "weights": weights.tolist(),
            # 约 12.69 和 15
            "weighted_information": (weights @ value).tolist(),
            "masked_weights": masked_weights.tolist(),
            # 10 和 15
            "masked_information": (masked_weights @ value).tolist(),
            "note": "教学用的手算 attention，不是训练后网络的真实注意力权重。"}


def demo_losses():
    """第 07 课：交叉熵、标签平滑，以及项目真实五项 loss 的计算。"""
    config, _ = smoke_model()
    # 人工构造一个 3 路局面的网络输出：10 个动作的 logits，其中动作 0 非法。
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]],
                          requires_grad=True)
    legal = torch.ones(1, 10, dtype=torch.bool)
    legal[0, 0] = False
    outputs = {"policy_logits": logits.masked_fill(~legal, float("-inf")),
               "value_logits": torch.zeros(1, 3), "score_mean": torch.zeros(1),
               "score_logvar": torch.zeros(1), "ownership_logits": torch.zeros(1, 3, 3, 3),
               "moves_left": torch.ones(1)}
    # 人工目标：策略在 9 个合法动作上均匀分布，胜负标签为“胜”（类别 2）。
    target = legal.float() / legal.sum(-1, keepdim=True)
    batch = {"legal": legal, "policy": target, "value": torch.tensor([2]),
             "score": torch.zeros(1), "ownership": torch.ones(1, 3, 3, dtype=torch.long),
             "moves_left": torch.ones(1), "aux_mask": torch.ones(1)}
    # 调用项目真实的 loss 函数
    result = losses(outputs, batch, config.train)
    result["total"].backward()
    toy_target = torch.tensor([0.75, 0.25])
    # 默认 0.25
    beta = config.train.policy_soft_weight
    return {
        # hard 标签的交叉熵 = -log(正确类别的预测概率)：概率越低，惩罚越大。
        "hard_win_ce": {str(p): -math.log(p) for p in (0.9, 0.5, 0.1)},
        # 软目标即使预测完全等于目标，交叉熵也不为 0（约 0.562）。
        "matched_soft_target_ce": float(-(toy_target * toy_target.log()).sum()),
        # 胜标签 [0,0,1] 平滑后约为 [0.0067,0.0067,0.9867]。
        "smoothed_win_target": _smoothed_one_hot(torch.tensor([2]), 3, 0.02).tolist(),
        # 中立点归属平滑后约为 [0.0807,0.8387,0.0807]。
        "smoothed_neutral_target": _ownership_targets(torch.tensor([1]), 0.02, 0.15).tolist(),
        # 策略损失中主目标与辅助软目标的实际比例：0.8 和 0.2。
        "policy_mix": [1 / (1 + beta), beta / (1 + beta)],
        "project_losses": {k: v.item() for k, v in result.items()},
        "finite_gradient": bool(torch.isfinite(logits.grad).all()),
        # 非法动作被 mask 后不会得到梯度。
        "illegal_gradient": logits.grad[0, 0].item(),
        "note": "人工目标只用于演示 loss 计算，不会写入 replay。"}


def demo_replay():
    """第 09 课：新鲜度加权抽样，以及 D8 旋转/翻转增强。"""
    # 两盘棋各 2 条记录：新盘每条权重 4，旧盘每条权重 1。
    weights = np.array([4.0, 4.0, 1.0, 1.0])
    n, action = 3, 1
    # 构造一条记录：策略目标全部集中在动作 1（第一行中间的点）。
    policy = np.zeros((1, n * n + 1), dtype=np.float32)
    policy[0, action] = 1
    batch = {"features": np.arange(21 * n * n, dtype=np.float32).reshape(1, 21, n, n),
             "legal": policy.astype(bool), "policy": policy,
             "ownership": np.ones((1, n, n), dtype=np.int64), "value": np.array([2]),
             "score": np.array([0.0]), "moves_left": np.array([1.0]), "aux_mask": np.array([1.0])}
    # 调用项目真实的增强函数，随机选择 8 种对称变换之一。
    transformed = augment(batch, np.random.default_rng(7))
    return {"position_weights": weights.tolist(),
            # 归一化后的抽样概率：[0.4,0.4,0.1,0.1]。
            "sampling_probability": (weights / weights.sum()).tolist(),
            # 棋盘旋转后，策略目标所在的点跟着移动。
            "action_before": action, "action_after": int(transformed["policy"][0].argmax()),
            # 合法 mask 与策略目标必须一起变换，始终对齐。
            "legal_policy_aligned": bool(np.array_equal(transformed["legal"],
                                                        transformed["policy"].astype(bool))),
            # pass 不在棋盘上，不随旋转改变。
            "pass_target": float(transformed["policy"][0, -1]),
            "note": "内存中的抽样与增强演示，不是真实训练数据文件。"}


def demo_train():
    """第 08、10、11 课：真实自博弈、训练、保存，再从存档恢复继续训练。"""
    config, model = smoke_model()
    device = torch.device("cpu")
    # 记下训练前的一组权重，用于比较
    before = model.stem.weight.detach().clone()
    # 所有文件都写在临时目录，with 块结束时自动删除。
    with TemporaryDirectory(prefix="transgo-ml-lesson-") as directory:
        root = Path(directory)
        # 1. 真实自博弈两盘。min_moves=4：前 4 手不主动 pass，让棋盘上先有几手棋。
        paths = generate(config, Evaluator(model, device), root / "replay", games=2,
                         report=lambda _event: None, score_utility=0.05, min_moves=4)
        arrays, _ = read_shard(paths[0])
        positions = sum(len(read_shard(path)[0]["value"]) for path in paths)
        output = root / "lesson.pt"
        # 2. 训练 2 步（2 次参数更新），完成后保存存档。
        first = train(config, model, device, root / "replay", output, steps=2,
                      report=lambda _event: None)
        difference = (model.stem.weight.detach() - before).abs()
        # 3. 从存档恢复模型、优化器和随机状态，再训练 1 步，step 应从 2 变成 3。
        restored, saved_config, payload = load_checkpoint(output, device, selfplay_only=True)
        second = train(saved_config, restored, device, root / "replay", output, steps=1,
                       resume=payload, report=lambda _event: None)
        _, _, resumed = load_checkpoint(output, device, selfplay_only=True)
        summaries = [first["summary"], second["summary"]]
        logged_steps = sum(len(Path(s["metrics_path"]).read_text().splitlines()) for s in summaries)
        result = {"games": len(paths), "positions": positions,
                  # 去重和窗口筛选后，实际进入训练的记录数。
                  "training_pool_positions": first["summary"]["positions"],
                  "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
                  # 第一条记录的目标：胜负类别、归一化目差、归一化剩余手数、策略概率总和。
                  "first_record_targets": {"value_class": int(arrays["value"][0]),
                                           "score": float(arrays["score"][0]),
                                           "moves_left": float(arrays["moves_left"][0]),
                                           "policy_sum": float(arrays["policy"][0].sum())},
                  "training_steps": first["summary"]["steps"], "saved_step": payload["step"],
                  "resumed_step": resumed["step"],
                  # 权重确实变了，只说明执行了更新，不代表棋力变强。
                  "weight_changed": bool(torch.any(difference > 0)),
                  "max_weight_change": difference.max().item(),
                  "loss_version": first["summary"]["loss_version"], "logged_steps": logged_steps,
                  "first_mean_losses": first["summary"]["mean"],
                  "note": "真实的 CPU 自博弈与训练；所有实验文件都是临时的。"}
    result["temporary_files_removed"] = not root.exists()
    return result


# 实验名到实验函数的对应表，--lesson 参数从这里选择。
LESSONS = {"linear": demo_linear, "tensors": demo_tensors, "probabilities": demo_probabilities,
           "gradient": demo_gradient, "forward": demo_forward, "attention": demo_attention,
           "losses": demo_losses, "replay": demo_replay, "train": demo_train}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lesson", choices=(*LESSONS, "all"), default="linear",
                        help="要运行的实验名；all 表示全部（默认 linear）")
    args = parser.parse_args(argv)
    selected = LESSONS if args.lesson == "all" else {args.lesson: LESSONS[args.lesson]}
    # 固定单线程，让小实验结果更稳定；结束后恢复原设置。
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name, experiment in selected.items():
            # 每个实验输出一段 JSON；ensure_ascii=False 让中文说明正常显示。
            print(json.dumps({"lesson": name, **experiment()}, indent=2,
                             ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
