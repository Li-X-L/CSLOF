# ARIMA.py
"""
基于ARIMA的离群点检测
"""
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import classification_report, confusion_matrix


class AutoARIMA():
    """
     基于自动选择参数的ARIMA离群点检测
     """

    def __init__(self, threshold = 3):
        """
        初始化参数

        Args:
            threshold: 残差离群阈值
        """
        self.threshold = threshold
        self.model = None
        self.residuals = None
        self.z_scores = None
        self.outliers = None

    def auto_arima_anomaly_detection(self, ts):
        """
        使用自动ARIMA参数选择的异常检测
        """
        # 自动选择ARIMA参数
        auto_model = auto_arima(ts, seasonal=False, stepwise=True,
                                suppress_warnings=True, error_action='ignore')

        # 使用自动选择的参数拟合模型
        self.model = ARIMA(ts, order=auto_model.order)
        model_fit = self.model.fit()

        # 基于残差检测异常
        self.residuals = np.array(model_fit.resid)
        residual_mean = np.mean(self.residuals)
        residual_std = np.std(self.residuals)

        z_scores = np.abs((self.residuals - residual_mean) / residual_std)

        self.z_scores = np.nan_to_num(z_scores)
        self.outliers = self.z_scores > self.threshold

        return self.outliers, self.z_scores


def plot_results(model, data, original_outliers):
        """
        可视化检测结果

        参数:
        - data: 原始数据
        - outliers: 检测到的离群点
        - original_outliers: 真实的离群点（如果有）
        - scores: 离群点得分
        """
        outliers = model.outliers
        scores = model.z_scores

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
            axes[1].axhline(y=model.threshold, color='r',
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

    # 建模、预测
    model_arima = AutoARIMA(threshold=3)
    predictions, decision_scores = model_arima.auto_arima_anomaly_detection(data)

    # 可视化结果
    plot_results(model_arima, data, true_outlier)

    # 评估结果
    print("\n检测结果统计:")
    print(f"检测到的异常点数量: {np.sum(predictions == 1)}")
    print(f"真实异常点数量: {np.sum(true_outlier == 1)}")

    print("\n分类报告:")
    print(classification_report(true_outlier, predictions))

    print("混淆矩阵:")
    print(confusion_matrix(true_outlier, predictions))


