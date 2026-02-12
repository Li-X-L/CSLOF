# FFT.py
"""
    基于FFT方法的离群点检测
    参考资料：Fourier Transform Based Spatial Outlier Mining
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


class FFT:
    """
    基于FFT方法的离群点检测
    参考资料：Fourier Transform Based Spatial Outlier Mining
    """
    def __init__(self, ifft_parameters=6, windows=4, threshold=2.0):
        """
        初始化参数

        Parameters:
        k: IFFT重构时使用的频率成分数量
        windows: 计算局部邻域时每边考虑的邻居数量
        z_threshold: Z值阈值
        """
        self.k = ifft_parameters
        self.c = windows//2
        self.z_threshold = threshold
        self.fitted_curve = None
        self.outlier = None
        self.scores = None


    def detect_outliers(self, data):
        """
        计算局部异常点
        """
        N = len(data)

        # 1. 傅里叶变换
        y = fft(data)

        # 2. 使用前k个频率成分进行逆傅里叶变换重构
        # 创建新的频率数组，只保留前k个成分
        y_reduced = np.zeros_like(y, dtype=complex)
        y_reduced[:self.k] = y[:self.k]
        y_reduced[-self.k + 1:] = y[-self.k + 1:] if self.k > 1 else y[-self.k:]

        # 逆傅里叶变换得到拟合曲线
        self.fitted_curve = np.real(ifft(y_reduced))

        # 3. 计算原始数据与拟合曲线的差异
        differences = np.abs(self.fitted_curve - data)
        mean_difference = np.mean(differences)

        # 4. 初步筛选候选异常点
        suspected_outliers = []
        suspected_indices = []
        local_differences = []

        for i in range(N):
            if differences[i] > mean_difference:
                # 计算局部邻域平均值
                left_neighbors = data[max(0, i - self.c):i]
                right_neighbors = data[i + 1:min(N, i + self.c + 1)]

                if len(left_neighbors) > 0 or len(right_neighbors) > 0:
                    neighbors = np.concatenate([left_neighbors, right_neighbors])
                    nav = np.mean(neighbors)

                    local_diff = data[i] - nav
                    suspected_outliers.append(data[i])
                    suspected_indices.append(i)
                    local_differences.append(local_diff)

        # 5. Z值检验确认异常点
        outliers = np.zeros(N)
        z_values = np.zeros(N)

        if len(local_differences) > 0:
            mean_local_diff = np.mean(local_differences)
            std_local_diff = np.std(local_differences)

            if std_local_diff > 0:  # 避免除零
                for i, (idx, local_diff) in enumerate(zip(suspected_indices, local_differences)):
                    z_value = (local_diff - mean_local_diff) / std_local_diff
                    z_values[idx] = z_value

                    if abs(z_value) > self.z_threshold:
                        outliers[idx] = 1

        self.outlier = outliers
        self.scores = z_values

        return self.outlier,  self.scores


def plot_results(model, data, original_outliers):
    """
    可视化检测结果

    参数:
    - model: FFT实例
    - data: 原始数据
    - original_outliers: 真实的离群点（如果有）
    """
    outliers, scores = model.outlier, model.scores

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制原始数据和检测到的离群点
    axes[0].plot(data, 'b-', label='Data', alpha=0.7, linewidth=1)
    outlier_indices = np.where(outliers == 1)[0]
    axes[0].scatter(outlier_indices, data[outlier_indices],
                    color='red', s=50, zorder=5, label='Predicted Outliers')
    axes[0].plot(model.fitted_curve, 'r--', label='fitted_curve', linewidth=1)

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
        axes[1].axhline(y=model.z_threshold, color='r', linestyle='--', label='Thresholds ')
        axes[1].axhline(y=-model.z_threshold, color='r', linestyle='--', label='Thresholds ')
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


# 运行示例
if __name__ == "__main__":

    # 生成示例数据
    data, true_outlier = generate_sample_data(n=200, outlier_ratio=0.05, seed=42)

    # 创建并训练模型
    fft_model = FFT(ifft_parameters=6, windows=6, threshold=2.0)

    # 检测异常
    # 预测
    predictions, decision_scores = fft_model.detect_outliers(data)

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
    plot_results(fft_model, data, true_outlier)