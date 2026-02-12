# Sub_IF.py
"""子序列孤立森林"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Any

class SubsequenceIsolationForest:
    """
    基于子序列的孤立森林离群点检测
    """

    def __init__(self, window_size=50, contamination : Any ='auto', n_estimators=100, random_state=42):
        """
        初始化参数

        Args:
            window_size: 滑动窗口大小
            contamination: 污染比例，即异常点的预期比例
            n_estimators: 孤立森林中树的数量
            random_state: 随机种子
        """
        self.window_size = window_size
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def create_subsequences(self, data):
        """
        创建子序列

        Args:
            data: 输入的时间序列数据

        Returns:
            subsequences: 子序列数组
        """
        subsequences = []
        n_samples = len(data)

        for i in range(n_samples - self.window_size + 1):
            subsequence = data[i:i + self.window_size]
            subsequences.append(subsequence)

        return np.array(subsequences)

    def fit_predict(self, data):
        """
        预测异常点

        Args:
            data: 测试数据

        Returns:
            anomaly_scores: 异常得分 (-1表示异常, 1表示正常)
            decision_scores: 决策函数值
        """

        # 创建子序列
        subsequences = self.create_subsequences(data)

        # 标准化特征
        subsequences_scaled = self.scaler.fit_transform(subsequences)

        # 训练孤立森林模型
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )

        # 预测
        anomaly_scores = self.model.fit_predict(subsequences_scaled)
        decision_scores = self.model.decision_function(subsequences_scaled)

        # 将子序列级别的预测扩展到原始数据点级别
        point_decision_scores = -self._expand_to_points(decision_scores, len(data))

        # 将决策得分转换为二进制标签
        anomaly_symbol = np.where(anomaly_scores == -1, 1, 0)
        true_contamination = sum(anomaly_symbol)/len(subsequences)
        threshold = np.percentile(point_decision_scores, (1-true_contamination)*100)
        outliers = np.where(point_decision_scores > threshold, 1, 0)

        return outliers, point_decision_scores

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
        axes[1].axhline(y=np.percentile(scores,  (1 - len(outlier_indices)/len(scores))*100), color='r', linestyle='--', label='Thresholds ')
        axes[1].set_title('Anomaly Decision Scores')
        axes[1].set_xlabel('Time Index')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    # 生成示例数据
    data, true_labels = generate_sample_data(n=400, outlier_ratio=0.05, seed=42)

    # 初始化模型
    sif = SubsequenceIsolationForest(
        window_size=5,
        contamination=0.05,
        n_estimators=100,
        random_state=42
    )

    # 预测
    predictions, decision_scores= sif.fit_predict(data)

    # 评估结果
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n检测结果统计:")
    print(f"检测到的异常点数量: {np.sum(predictions == 1)}")
    print(f"真实异常点数量: {np.sum(true_labels == 1)}")

    print("\n分类报告:")
    print(classification_report(true_labels, predictions))

    print("混淆矩阵:")
    print(confusion_matrix(true_labels, predictions))

    # 可视化结果
    print("生成可视化图表...")
    plot_results(data, predictions, true_labels, decision_scores)