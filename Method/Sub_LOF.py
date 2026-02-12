# Sub_LOF.py
"""子序列LOF"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from typing import Any

class SubsequenceLOF:
    """
    基于子序列的LOF离群点检测
    """

    def __init__(self, window_size=10, step_size: int =1, contamination: Any ='auto', n_neighbors: int =20):
        """
        初始化参数

        Args:
            window_size: 滑动窗口大小
            step_size: 滑动步长
            contamination: 污染比例，用于LOF算法
            n_neighbors: LOF算法中的邻居数量
        """
        self.window_size = window_size
        self.step_size = step_size
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )
        self.subsequences = None
        self.lof_scores = None
        self.outlier = None

    def create_subsequences(self, time_series):
        """
        从时间序列创建子序列

        Args:
            time_series: 输入时间序列

        Returns:
            子序列矩阵
        """
        n = len(time_series)
        subsequences = []
        start_indices = []

        for i in range(0, n - self.window_size + 1, self.step_size):
            subsequence = time_series[i:i + self.window_size]
            subsequences.append(subsequence)
            start_indices.append(i)

        self.subsequences = np.array(subsequences)

        return self.subsequences

    def fit_predict(self, time_series):
        """
        训练模型并预测离群点

        Args:
            time_series: 输入时间序列

        Returns:
            离群点标签 (-1表示离群点，1表示正常点)
        """
        # 创建子序列
        subsequences = self.create_subsequences(time_series)

        # 计算LOF得分
        lof_labels = self.lof.fit_predict(subsequences)
        lof_scores = -self.lof.negative_outlier_factor_  # 转换为正数，值越大越异常

        # 将子序列级别的预测扩展到原始数据点级别
        point_decision_scores = self._expand_to_points(lof_scores, len(time_series))
        self.lof_scores = point_decision_scores

        # 将决策得分转换为二进制标签
        anomaly_symbol = np.where(lof_labels == -1, 1, 0)
        true_contamination = sum(anomaly_symbol)/len(subsequences)
        threshold = np.percentile(point_decision_scores, (1-true_contamination)*100)
        self.outlier = np.where(point_decision_scores > threshold, 1, 0)

        return self.outlier, self.lof_scores

    def _expand_to_points(self, subsequence_scores, original_length):
        """
        将子序列级别的得分扩展到原始数据点级别

        Args:
            subsequence_scores: 子序列级别的得分
            original_length: 原始数据长度

        Returns:
            point_scores: 数据点级别的得分
        """
        point_scores = np.zeros(original_length)
        point_counts = np.zeros(original_length)

        for i, score in enumerate(subsequence_scores):
            start_idx = i
            end_idx = i + self.window_size
            point_scores[start_idx:end_idx] += score
            point_counts[start_idx:end_idx] += 1

        # 避免除零
        point_counts[point_counts == 0] = 1
        point_scores = point_scores / point_counts

        return point_scores

    def get_outliers(self):
        """
        获取异常段

        Returns:
            异常段的起始索引列表
        """
        return self.outlier

    def get_lof_scores(self):
        """
        获取LOF得分

        Returns:
            LOF得分数组
        """
        return self.lof_scores

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
        axes[0].plot(data, 'b-', label='Data', alpha=0.7, linewidth=1)
        outlier_indices = np.where(outliers == 1)[0]
        axes[0].scatter(outlier_indices, data[outlier_indices],
                        color='red', s=50, zorder=5, label='Predicted Outliers')

        # 如果有真实离群点，用不同颜色标记
        original_outliers_indices = np.where(original_outliers == 1)[0]
        if original_outliers_indices is not None:
            axes[0].scatter(original_outliers_indices, data[original_outliers_indices],
                            color='orange', s=30, zorder=6, marker='x',
                            label='True Outliers', linewidth=2)

        axes[0].set_title('Time Series with Outliers Highlighted')
        axes[0].set_xlabel('Time Index')
        # axes[0].set_ylabel('Num')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 绘制离群点得分
        if scores is not None:
            axes[1].plot(scores, 'g-', alpha=0.7, label='Scores')
            axes[1].axhline(y=np.percentile(scores, (1 - len(outlier_indices) / len(scores)) * 100), color='r',
                            linestyle='--', label='Thresholds ')
            axes[1].set_title('Anomaly Decision Scores')
            axes[1].set_xlabel('Time Index')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

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


# 测试示例
if __name__ == "__main__":

    # 生成示例数据
    data, true_outlier = generate_sample_data(n=200, outlier_ratio=0.05, seed=42)

    # 创建并训练模型
    detector = SubsequenceLOF(
        window_size=5,
        step_size=1,
        contamination=0.1,
        n_neighbors=20
    )

    # 检测异常
    # 预测
    predictions, decision_scores = detector.fit_predict(data)

    # 评估结果
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n检测结果统计:")
    print(f"检测到的异常点数量: {np.sum(predictions == 1)}")
    print(f"真实异常点数量: {np.sum(true_outlier == 1)}")

    print("\n分类报告:")
    print(classification_report(true_outlier, predictions))

    print("混淆矩阵:")
    print(confusion_matrix(true_outlier, predictions))

    # 可视化结果
    print("生成可视化图表...")
    plot_results(data, predictions, true_outlier, decision_scores)

