# PS_SVM.py
"""
基于相空间一类支持向量机进行离群点检测
参考论文：Time-series Novelty Detection Using One-class Support Vector Machines
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class PSSVM:
    """
    基于单类SVM的时间序列异常检测
    参考论文: "Time-series Novelty Detection Using One-class Support Vector Machines"
    """

    def __init__(self, embedding_dims=None, nu=0.05, gamma='scale', use_projected_space=True):
        """
        初始化参数

        Parameters:
        -----------
        embedding_dims : list, 嵌入维度列表，如 [3, 5, 7, 9, 11]
        nu : float, 单类SVM参数，异常点比例上限
        gamma : str or float, RBF核参数
        use_projected_space : bool, 是否使用投影相空间
        """
        self.embedding_dims = embedding_dims if embedding_dims else [3, 5, 7, 9, 11]
        self.nu = nu
        self.gamma = gamma
        self.use_projected_space = use_projected_space
        self.models = {}  # 存储不同嵌入维度的模型

    def _create_phase_space(self, time_series, E):
        """
        将时间序列转换为相空间向量

        Parameters:
        -----------
        time_series : array-like, 时间序列数据
        E: int, 嵌入维度

        Returns:
        --------
        vectors : ndarray, 相空间向量
        indices : ndarray, 每个向量对应的时间序列索引
        """
        n = len(time_series)
        if n < E:
            raise ValueError(f"时间序列长度 {n} 小于嵌入维度 {E}")

        vectors = []
        indices = []

        for t in range(E - 1, n):
            vector = time_series[t - E + 1:t + 1]
            vectors.append(vector)
            indices.append(t)

        return np.array(vectors), np.array(indices)

    def _project_to_orthogonal_space(self, vectors):
        """
        将相空间向量投影到与对角线垂直的子空间

        Parameters:
        -----------
        vectors : ndarray, 相空间向量

        Returns:
        --------
        projected_vectors : ndarray, 投影后的向量
        """
        E = vectors.shape[1]

        # 创建对角线方向的单位向量
        diagonal_vector = np.ones(E) / np.sqrt(E)

        # 投影矩阵: I - vv^T
        projection_matrix = np.eye(E) - np.outer(diagonal_vector, diagonal_vector)

        # 应用投影
        projected_vectors = vectors @ projection_matrix.T

        return projected_vectors

    def fit_predict(self, time_series):
        """
        在正常时间序列上训练模型

        Parameters:
        -----------
        time_series : array-like, 正常时间序列数据
        """
        time_series = np.array(time_series).flatten()
        n = len(time_series)

        # 初始化结果数组
        individual_results = []
        score_list = []

        for E in self.embedding_dims:

            # 创建相空间向量
            vectors, indices = self._create_phase_space(time_series, E)

            # 可选：投影到正交空间
            if self.use_projected_space:
                vectors = self._project_to_orthogonal_space(vectors)

            # 标准化
            scaler = StandardScaler()
            vectors_scaled = scaler.fit_transform(vectors)

            # 预测
            self.models[E] = OneClassSVM(nu=self.nu, kernel='rbf', gamma=self.gamma)
            predictions = self.models[E].fit_predict(vectors_scaled)
            vector_scores = -self.models[E].decision_function(vectors_scaled)

            # 将异常分数映射到点
            point_scores = np.full(time_series.shape, np.nan)
            point_scores[: -E + 1] = vector_scores
            score_list.append(point_scores)

            # 将预测结果转换为 0(正常)/1(异常)
            anomaly_flags = np.where(predictions == -1, 1, 0)

            # 将向量级别的异常标记映射回时间点级别
            point_anomalies = np.zeros(n, dtype=bool)
            for i, idx in enumerate(indices):
                if anomaly_flags[i] == 1:
                    # 标记该向量对应的所有时间点为异常
                    start_idx = max(0, idx - E + 1)
                    point_anomalies[start_idx : idx + 1] = 1

            individual_results.append(point_anomalies)

        # 多尺度融合：只有当所有维度都认为是异常时才标记为异常
        outliers = np.all(individual_results, axis=0)
        scores = np.nansum(np.array(score_list), axis=0)

        return outliers, scores


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
        axes[1].set_title('Anomaly Decision Scores')
        axes[1].set_xlabel('Time Index')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 演示示例
if __name__ == "__main__":

    # 生成合成数据
    data, true_outlier = generate_sample_data(n=200, outlier_ratio=0.05, seed=42)

    # 创建检测器
    detector = PSSVM(
        embedding_dims=[3,5,7,9,11],
        nu=0.1,
        gamma='scale',
        use_projected_space=True
    )

    # 建模、预测
    detected_outliers, scores = detector.fit_predict(data)

    # 评估结果
    from sklearn.metrics import classification_report, confusion_matrix

    print("\n检测结果统计:")
    print(f"检测到的异常点数量: {np.sum(detected_outliers == 1)}")
    print(f"真实异常点数量: {np.sum(true_outlier == 1)}")

    print("\n分类报告:")
    print(classification_report(true_outlier, detected_outliers))

    print("混淆矩阵:")
    print(confusion_matrix(true_outlier, detected_outliers))

    # 可视化结果
    print("生成可视化图表...")
    plot_results(data, detected_outliers, true_outlier, scores)