# LOF.py
"""单变量数据LOF离群点检测"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class LOF:
    """
    基于子序列的LOF离群点检测
    """

    def __init__(self, contamination ='auto', n_neighbors: int =20):
        """
        初始化参数

        Args:
            contamination: 污染比例，用于LOF算法
            n_neighbors: LOF算法中的邻居数量
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.model = None
        self.lof_scores = None
        self.outlier = None

    def fit_predict(self, time_series):
        """
        训练模型并预测离群点

        Args:
            time_series: 输入时间序列

        Returns:
            离群点标签 (1表示离群点，0表示正常点)
        """

        self.model = LocalOutlierFactor(
            n_neighbors = self.n_neighbors,
            contamination = self.contamination
        )
        y_pred = self.model.fit_predict(time_series)  # 得到预测标签 (1正常, -1异常)

        # 标签（0正常, 1异常）、异常分数
        self.outlier, self.lof_scores = np.where(y_pred == -1, 1, 0), -self.model.negative_outlier_factor_

        return self.outlier, self.lof_scores


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


if __name__ == "__main__":

    # 生成示例数据
    data, true_outlier = generate_sample_data(n=200, outlier_ratio=0.05, seed=42)

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    # 建模、预测
    lof = LOF(contamination='auto', n_neighbors=20)
    predictions, decision_scores = lof.fit_predict(data_scaled)

    # 评估结果
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    print("\n检测结果统计:")
    print(f"检测到的异常点数量: {np.sum(predictions == 1)}")
    print(f"真实异常点数量: {np.sum(true_outlier == 1)}")

    print("\n分类报告:")
    print(classification_report(true_outlier, predictions))

    print("AUC值:")
    print(roc_auc_score(true_outlier, predictions))

    print("\n混淆矩阵:")
    print(confusion_matrix(true_outlier, predictions))

    # 可视化结果
    print("生成可视化图表...")
    plot_results(data, predictions, true_outlier, decision_scores)
