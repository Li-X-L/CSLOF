# SR.py
"""
基于Spectral Residual 的异常检测
参考论文：“Time-Series Anomaly Detection Service at Microsoft”
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import convolve2d
from sklearn.metrics import classification_report, confusion_matrix


class SR:
    def __init__(self, window_size=1440, q=3, z=21, tau=3, kappa=5, m=5):
        """
        Spectral Residual （SR）异常检测模型

        参数:
        - window_size: 滑动窗口大小
        - q: 卷积核大小，用于计算平均谱
        - z: 局部平均的前置点数
        - tau: 异常检测阈值
        - kappa: 估计点数量
        - m: 计算梯度时考虑的前置点数
        """
        self.window_size = window_size
        self.q = q
        self.z = z
        self.tau = tau
        self.kappa = kappa
        self.m = m

    def _add_estimated_points(self, sequence):
        """
        在序列末尾添加估计点，使当前点位于窗口中心
        """
        n = len(sequence)

        # 计算平均梯度
        gradients = []
        for l in range(1, self.m + 1):
            if n - l - 1 >= 0:
                grad = (sequence[-1] - sequence[n - l - 1]) / l
                gradients.append(grad)

        if gradients:
            avg_gradient = np.mean(gradients)
        else:
            avg_gradient = 0

        # 生成估计点
        estimated_points = []
        for i in range(1, self.kappa + 1):
            if n - self.m >= 0:
                estimated_point = sequence[n - self.m] + avg_gradient * self.m
            else:
                estimated_point = sequence[-1] + avg_gradient * i
            estimated_points.append(estimated_point)

        # 将估计点添加到原序列
        extended_sequence = np.concatenate([sequence, estimated_points])
        return extended_sequence

    def _create_convolution_matrix(self, size):
        """创建卷积矩阵 h_q(f)"""
        return np.ones((size, size)) / (size ** 2)

    def spectral_residual(self, sequence):
        """
        计算序列的 Spectral Residual 和显著性图

        参数:
        - sequence: 输入时间序列

        返回:
        - saliency_map: 显著性图
        - extended_sequence: 扩展后的序列（包含估计点）
        """
        # 添加估计点
        extended_sequence = self._add_estimated_points(sequence)

        # 步骤1: 傅里叶变换
        fft_result = fft(extended_sequence)
        amplitude = np.abs(fft_result)  # 振幅谱 A(f)
        phase = np.angle(fft_result)  # 相位谱 P(f)

        # 步骤2: 计算对数振幅谱
        log_amplitude = np.log(amplitude + 1e-8)  # 避免log(0)

        # 步骤3: 计算平均谱 AL(f)
        # 将一维序列转换为二维矩阵进行卷积
        log_amp_2d = log_amplitude.reshape(-1, 1)
        h_matrix = self._create_convolution_matrix(self.q)

        # 使用边界扩展进行卷积
        avg_log_amplitude = convolve2d(log_amp_2d, h_matrix, mode='same', boundary='symm')
        avg_log_amplitude = avg_log_amplitude.flatten()

        # 步骤4: 计算谱残差 R(f)
        spectral_residual = log_amplitude - avg_log_amplitude

        # 步骤5: 逆傅里叶变换得到显著性图
        # 重建复数信号: exp(R(f) + i*P(f))
        complex_signal = np.exp(spectral_residual + 1j * phase)
        saliency_map = np.abs(ifft(complex_signal))

        # 只返回原始序列长度对应的显著性图
        saliency_map_original = saliency_map[:len(sequence)]

        return saliency_map_original, extended_sequence

    def detect_anomalies(self, sequence):
        """
        检测序列中的异常点

        参数:
        - sequence: 输入时间序列

        返回:
        - anomalies: 异常点标记 (1表示异常, 0表示正常)
        - saliency_map: 显著性图
        """
        if len(sequence) < self.z:
            raise ValueError(f"序列长度必须至少为 {self.z}")

        saliency_map, _ = self.spectral_residual(sequence)
        anomaly_score = np.zeros(len(sequence))

        for i in range(len(sequence)):
            if i < self.z:
                # 对于前z个点，使用可用的点计算局部平均
                local_avg = np.mean(saliency_map[:i + 1])
            else:
                local_avg = np.mean(saliency_map[i - self.z:i])

            # 计算异常分数
            if local_avg > 0:
                anomaly_score[i] = (saliency_map[i] - local_avg) / local_avg
            else:
                anomaly_score[i] = 0

        # 判断是否为异常
        anomalies = np.where(anomaly_score > self.tau, 1, 0)

        return anomalies, anomaly_score

def _plot_results(data, outliers, original_outliers=None, scores=None):
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
            axes[1].axhline(y=sr_model.tau, color='red', linestyle='--', label=f'Thresholds (τ={sr_model.tau})')
            axes[1].set_title('Anomaly Decision Scores')
            axes[1].set_xlabel('Time Index')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def _generate_sample_data(n=200, outlier_ratio=0.05, seed=42):
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
    data, true_outlier = _generate_sample_data(n=200, outlier_ratio=0.05, seed=42)

    # 初始化模型
    sr_model = SR(window_size=100, q=3, z=21, tau=3, kappa=5, m=5)

    # 检测异常
    predictions, decision_scores = sr_model.detect_anomalies(data)

    # 评估结果

    print("\n检测结果统计:")
    print(f"检测到的异常点数量: {np.sum(predictions == 1)}")
    print(f"真实异常点数量: {np.sum(true_outlier == 1)}")

    print("\n分类报告:")
    print(classification_report(true_outlier, predictions))

    print("混淆矩阵:")
    print(confusion_matrix(true_outlier, predictions))

    # 可视化结果
    print("生成可视化图表...")
    _plot_results(data, predictions, true_outlier, decision_scores)