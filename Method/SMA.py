# SMA.py
"""
基于简单移动平均进行离群点检测
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Union


class SMAOutlierDetector:
    """
    基于简单移动平均的离群点检测器
    """

    def __init__(self, window_size: int = 5, threshold: float = 2.0):
        """
        初始化检测器

        参数:
            window_size: 移动平均窗口大小
            threshold: 离群点检测阈值（标准差的倍数）
        """
        self.window_size = window_size
        self.threshold = threshold
        self.sma_values = None
        self.residuals = None
        self.std_residual = None

    def _calculate_sma(self, data: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """
        计算简单移动平均

        参数:
            data: 输入数据序列

        返回:
            sma_values: 移动平均值序列
        """
        series = pd.Series(data)
        sma_values = series.rolling(window=self.window_size, center=True).mean()

        # 处理边界值（使用较小的窗口）
        for i in range(self.window_size // 2):
            if pd.isna(sma_values.iloc[i]):
                sma_values.iloc[i] = series.iloc[:i + self.window_size // 2 + 1].mean()
            if pd.isna(sma_values.iloc[-(i + 1)]):
                sma_values.iloc[-(i + 1)] = series.iloc[-(i + self.window_size // 2 + 1):].mean()

        self.sma_values = sma_values.values
        return self.sma_values

    def detect_outliers(self, data: Union[List, np.ndarray, pd.Series]) -> np.ndarray:
        """
        检测离群点

        参数:
            data: 输入数据序列

        返回:
            outliers: 布尔数组，True表示离群点
        """
        data = np.array(data)

        # 计算移动平均
        sma = self._calculate_sma(data)

        # 计算残差（实际值与移动平均的差值）
        self.residuals = data - sma

        # 计算残差的标准差
        self.std_residual = np.std(self.residuals)

        # 检测离群点（残差超过阈值倍标准差）
        outlier_mask = np.where(np.abs(self.residuals) > (self.threshold * self.std_residual), 1, 0)

        return outlier_mask

    def plot_detection(self, data: Union[List, np.ndarray, pd.Series],
                       outliers: np.ndarray = None,
                       title: str = "SMA离群点检测") -> None:
        """
        可视化检测结果

        参数:
            data: 输入数据序列
            outliers: 离群点掩码（如果为None，则自动检测）
            title: 图表标题
        """
        # 支持中文
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        data = np.array(data)

        if outliers is None:
            outliers = self.detect_outliers(data)

        plt.figure(figsize=(12, 8))

        # 绘制原始数据
        plt.subplot(2, 1, 1)
        plt.plot(data, 'b-', label='原始数据', alpha=0.7, linewidth=1)
        plt.plot(self.sma_values, 'g-', label=f'SMA (窗口={self.window_size})', linewidth=2)
        plt.scatter(np.where(outliers)[0], data[np.where(outliers)[0]],
                    color='red', s=50, zorder=5, label='离群点')
        plt.legend()
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # 绘制残差
        plt.subplot(2, 1, 2)
        plt.plot(self.residuals, 'orange', label='残差', alpha=0.7)
        plt.axhline(y=self.threshold * self.std_residual, color='r',
                    linestyle='--', label=f'阈值 (±{self.threshold}σ)')
        plt.axhline(y=-self.threshold * self.std_residual, color='r', linestyle='--')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.scatter(np.where(outliers)[0], self.residuals[np.where(outliers)[0]],
                    color='red', s=50, zorder=5, label='离群残差')
        plt.legend()
        plt.title('残差分析')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


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


# 示例使用
if __name__ == "__main__":

    # 生成示例数据
    data, true_outlier = generate_sample_data(n=200, outlier_ratio=0.05, seed=42)

    # 创建检测器
    detector = SMAOutlierDetector(window_size=10, threshold=2.5)

    # 检测离群点
    outliers = detector.detect_outliers(data)

    # 可视化结果
    detector.plot_detection(data, outliers, "SMA离群点检测示例")

    # 显示离群点详细信息
    outlier_indices = np.where(outliers)[0]
    print(f"\n检测到的离群点索引: {outlier_indices}")
    print(f"离群点对应的值: {data[outlier_indices]}")

    # 参数敏感性分析
    print("\n=== 参数敏感性分析 ===")
    window_sizes = [5, 10, 15]
    thresholds = [2.0, 2.5, 3.0]

    for window in window_sizes:
        for threshold in thresholds:
            temp_detector = SMAOutlierDetector(window_size=window, threshold=threshold)
            temp_outliers = temp_detector.detect_outliers(data)
            print(f"窗口={window}, 阈值={threshold}: 检测到{sum(temp_outliers)}个离群点")