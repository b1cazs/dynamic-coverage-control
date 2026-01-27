import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_curve(exp_dir, key="reward"):
    """
    从 train_curve.npy 中提取指定指标（如 reward）
    兼容：
    1) ndarray of dict
    2) dict of list
    3) list
    4) ndarray of float
    """
    path = os.path.join(exp_dir, "train_curve.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    data = np.load(path, allow_pickle=True)

    # 情况 1：ndarray，元素是 dict（最常见）
    if isinstance(data, np.ndarray) and len(data) > 0 and isinstance(data[0], dict):
        if key not in data[0]:
            raise KeyError(f"{key} not found in train_curve elements")
        return np.array([step[key] for step in data], dtype=float)

    # 情况 2：dict，key 对应 list
    if isinstance(data, dict):
        if key not in data:
            raise KeyError(f"{key} not found in train_curve dict")
        return np.array(data[key], dtype=float)

    # 情况 3：普通 list
    if isinstance(data, list):
        return np.array(data, dtype=float)

    # 情况 4：已经是数值 ndarray
    return np.array(data, dtype=float)


def plot_compare(curves, title, save_name):
    """
    curves: dict {label: np.array}
    """
    plt.figure()
    for label, data in curves.items():
        plt.plot(data, label=label)

    plt.xlabel("Episode") # 横轴（x轴），含义：训练轮次（episode index）
    plt.ylabel("Average Reward") # 纵轴（y轴），含义：该 episode 内，所有无人机的平均 reward，是一次完整任务的整体性能评价
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


def stat_last_n(data, n=100):
    tail = data[-n:]
    return tail.mean(), tail.std()


if __name__ == "__main__":

    # ========= 原实验与优化后实验路径 =========
    BASELINE_DIR = "results/uav_dcc/0114_1438_sd0"
    MULTIOBJ_DIR = "results/uav_dcc/0127_1438_sd0"

    # ========= 加载数据 =========
    curve_base = load_curve(BASELINE_DIR)
    curve_new  = load_curve(MULTIOBJ_DIR)

    # ========= 画对比曲线 =========
    plot_compare(
        {
            "Baseline": curve_base,
            "Multi-objective Reward": curve_new
        },
        title="Training Reward Curve Comparison",
        save_name="reward_curve_compare.png"
    )

    # ========= 统计指标 =========
    N = 100
    mean_b, std_b = stat_last_n(curve_base, N)
    mean_n, std_n = stat_last_n(curve_new, N)

    df = pd.DataFrame({
        "Method": ["Baseline", "Multi-objective"],
        "Reward Mean": [mean_b, mean_n], # 平均奖励，训练后期（最后 100 个 episode）智能体获得的平均回报
        "Reward Std": [std_b, std_n] # 奖励标准差，训练后期 reward 的波动程度（小：策略稳定，大：策略不稳定/震荡）
    })

    print("\n===== Reward Statistics (Last {} Episodes) =====".format(N))
    print(df)

    df.to_csv("reward_statistics.csv", index=False)