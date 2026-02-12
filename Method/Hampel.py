# Hampel.py
"""基于Hampel标识符法进行离群点检测"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置全局字体和字号
# 支持中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# plt.rcParams['font.family'] = 'Times New Roman'  # 设置中文字体
# plt.rcParams['font.size'] = 10          # 设置全局字号

class HampelDetector:
    """
    Hampel标识符法离群点检测器

    基于中位数和MAD（中位数绝对偏差）的鲁棒离群点检测方法
    """

    def __init__(self, window_size=5, n_sigma=3.0, min_periods=None):
        """
        初始化Hampel检测器

        参数:
        - window_size: 滑动窗口大小
        - n_sigma: 离群点判断的sigma倍数
        - min_periods: 最小观测数，默认为window_size
        """
        self.window_size = window_size
        self.n_sigma = n_sigma
        self.min_periods = min_periods if min_periods is not None else window_size

    def _mad(self, data):
        """
        计算中位数绝对偏差 (MAD)

        参数:
        - data: 输入数据

        返回:
        - MAD值
        """
        median = np.median(data)
        deviations = np.abs(data - median)
        return np.median(deviations)

    def detect(self, data):
        """
        检测离群点

        参数:
        - data: 输入数据，可以是list, numpy array或pandas Series

        返回:
        - outliers: 离群点布尔数组
        - scores: 离群点得分
        """
        data = np.array(data)
        n = len(data)
        outliers = np.zeros(n, dtype=int)
        hampel_values = np.full(n, np.nan)

        for i in range(n):
            # 确定滑动窗口边界
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(n, i + self.window_size // 2 + 1)

            window_data = data[start_idx:end_idx]

            # 确保窗口内有足够的数据点
            if len(window_data) >= self.min_periods:
                median = np.median(window_data)
                mad = self._mad(window_data)

                # 计算Hampel值（标准化得分）
                if mad > 0:
                    hampel_value = np.abs(data[i] - median) / (1.4826 * mad)
                    hampel_values[i] = hampel_value

                    # 判断是否为离群点
                    if hampel_value > self.n_sigma:
                        outliers[i] = 1

        hampel_values = np.nan_to_num(hampel_values)

        return outliers, hampel_values


def generate_sample_data(n=200, outlier_ratio=0.05, seed=42):
    """
    生成包含离群点的示例数据

    参数:
    - n: 数据点数量
    - outlier_ratio: 离群点比例
    - seed: 随机种子

    返回:
    - 生成的数据
    """
    np.random.seed(seed)

    # 生成正常数据（正弦波 + 噪声）
    t = np.linspace(0, 4 * np.pi, n)
    normal_data = 10 * np.sin(t) + np.random.normal(0, 1, n)

    # 添加离群点
    n_outliers = int(n * outlier_ratio)
    outlier_indices = np.random.choice(n, n_outliers, replace=False)

    data_with_outliers = normal_data.copy()
    labels = np.zeros(n)

    for idx in outlier_indices:
        # 随机生成较大的偏差
        magnitude = np.random.uniform(4, 8)
        direction = np.random.choice([-1, 1])
        data_with_outliers[idx] += direction * magnitude
        labels[idx] = 1

    return data_with_outliers, labels


def plot_results(data, outliers, original_outliers=None, scores=None):
    """
    可视化检测结果

    参数:
    - data: 原始数据
    - outliers: 检测到的离群点
    - original_outliers: 真实的离群点（如果有）
    - scores: 离群点得分
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制原始数据和检测到的离群点
    axes[0].plot(data, 'b-', label='正常数据', alpha=0.7, linewidth=1)
    outlier_indices = np.where(outliers)[0]
    axes[0].scatter(outlier_indices, data[outlier_indices],
                    color='red', s=50, zorder=5, label='检测到的离群点')

    # 如果有真实离群点，用不同颜色标记
    original_outliers_indices = np.where(original_outliers)[0]
    if original_outliers_indices is not None:
        axes[0].scatter(original_outliers_indices, data[original_outliers_indices],
                        color='orange', s=30, zorder=6, marker='x',
                        label='真实离群点', linewidth=2)

    axes[0].set_title('Hampel离群点检测结果')
    axes[0].set_xlabel('时间/索引')
    axes[0].set_ylabel('数值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 绘制离群点得分
    if scores is not None:
        axes[1].plot(scores, 'g-', alpha=0.7, label='Hampel得分')
        axes[1].axhline(y=3, color='r', linestyle='--', label='阈值 (3σ)')
        axes[1].set_title('离群点得分')
        axes[1].set_xlabel('时间/索引')
        axes[1].set_ylabel('Hampel值')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_performance(true_outliers, detected_outliers, n_total):
    """
    评估检测性能

    参数:
    - true_outliers: 真实离群点
    - detected_outliers: 检测到的离群点
    - n_total: 总数据点数

    返回:
    - 性能指标字典
    """
    true_outliers_set = set(true_outliers)
    detected_outliers_set = set(np.where(detected_outliers)[0])

    # 计算各种指标
    true_positive = len(true_outliers_set & detected_outliers_set)
    false_positive = len(detected_outliers_set - true_outliers_set)
    false_negative = len(true_outliers_set - detected_outliers_set)
    true_negative = n_total - true_positive - false_positive - false_negative

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positive + true_negative) / n_total

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative
    }


# 示例使用
if __name__ == "__main__":

    # 1. 生成示例数据
    data, true_outlier_indices = generate_sample_data(n=200, outlier_ratio=0.08)
    print(f"   生成数据点: {len(data)}")
    print(f"   真实离群点数量: {sum(true_outlier_indices)}")

    # 2. 使用Hampel检测器
    detector = HampelDetector(window_size=7, n_sigma=3.0)

    # 方法1: 基本方法
    outliers_basic, hampel_values = detector.detect(data)
    print(f"   基本方法检测到离群点: {np.sum(outliers_basic)}")

    # 3. 评估性能
    metrics_basic = evaluate_performance(true_outlier_indices, outliers_basic, len(data))

    print("   基本方法:")
    print(f"     - precision: {metrics_basic['precision']:.3f}", end=',')
    print(f"     - recall: {metrics_basic['recall']:.3f}", end=',')
    print(f"     - f1_score: {metrics_basic['f1_score']:.3f}")

    # 4. 可视化结果
    print("\n4. 生成可视化图表...")
    plot_results(data, outliers_basic, true_outlier_indices, hampel_values)

    # 5. 参数调优示例
    print("\n5. 不同参数对比:")
    window_sizes = [5, 7, 9]
    sigma_thresholds = [2.5, 3.0, 3.5]

    best_f1 = 0
    best_params = {}

    for window in window_sizes:
        for sigma in sigma_thresholds:
            detector_tune = HampelDetector(window_size=window, n_sigma=sigma)
            outliers_tune, _ = detector_tune.detect(data)
            metrics_tune = evaluate_performance(true_outlier_indices, outliers_tune, len(data))

            if metrics_tune['f1_score'] > best_f1:
                best_f1 = metrics_tune['f1_score']
                best_params = {'window_size': window, 'n_sigma': sigma}

            print(f"   窗口大小={window}, σ={sigma}: F1={metrics_tune['f1_score']:.3f}")

    print(f"\n   最佳参数: {best_params}, F1分数: {best_f1:.3f}")

    # 6. 实际应用建议
    print("\n6. 实际应用建议:")
    print("   - 对于平稳时间序列，建议使用较小的窗口大小 (5-7)")
    print("   - 对于波动较大的数据，建议使用较大的窗口大小 (9-15)")
    print("   - 阈值通常设置在2.5-3.5之间，根据误报率需求调整")