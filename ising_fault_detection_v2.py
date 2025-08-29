#!/usr/bin/env python3
"""
改进版伊辛模型故障检测 - 解决概率预测单一化问题
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')


class ImprovedIsingFaultDetector:
    """改进版伊辛模型故障检测器"""
    
    def __init__(self, n_sensors=354, alpha=0.5, beta=0.01, lambda_reg=0.1, 
                 initial_temp=2.0, final_temp=0.5, max_iter=100, mc_steps=50):
        """初始化改进版伊辛模型"""
        self.n_sensors = n_sensors
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.max_iter = max_iter
        self.mc_steps = mc_steps
        
        # 温度调度
        self.temperature = initial_temp
        
        # 初始化模型参数
        self._initialize_parameters()
        
        # 训练历史
        self.loss_history = []
        self.energy_history = []
        self.ce_loss_history = []
        self.accuracy_history = []
        self.temperature_history = []
        
        # 数据预处理器
        self.scaler = StandardScaler()  # 改用标准化
        
    def _initialize_parameters(self):
        """改进的参数初始化"""
        # 相互作用参数 J_ij - 基于传感器相关性初始化
        self.J = np.zeros((self.n_sensors, self.n_sensors))
        
        # 外部场参数 h_i - 根据传感器类型初始化
        self.h = np.zeros(self.n_sensors)
        
        # 自旋状态 - 初始化为正常状态
        self.spins = np.ones(self.n_sensors)
        
        print(f"参数初始化完成:")
        print(f"  - J矩阵形状: {self.J.shape}")
        print(f"  - h向量长度: {len(self.h)}")
        print(f"  - 初始自旋状态: 全部正常")
        
    def preprocess_data(self, X, y=None):
        """改进的数据预处理"""
        print("开始数据预处理...")
        
        # 1. 标准化处理
        if y is not None:
            X_normalized = self.scaler.fit_transform(X)
            print(f"  - 训练集标准化完成，均值: {np.mean(X_normalized):.3f}, 标准差: {np.std(X_normalized):.3f}")
        else:
            X_normalized = self.scaler.transform(X)
            print(f"  - 测试集标准化完成")
        
        # 2. 计算传感器读数与期望值的差值
        expected_values = np.zeros(self.n_sensors)  # 标准化后期望值为0
        d_values = X_normalized - expected_values
        
        print(f"  - 特征数量: {X_normalized.shape[1]}")
        print(f"  - 样本数量: {X_normalized.shape[0]}")
        print(f"  - d值统计: mean={np.mean(d_values):.3f}, std={np.std(d_values):.3f}")
        
        return X_normalized, d_values
    
    def compute_energy(self, spins, d_sample):
        """计算能量函数"""
        # 相互作用项
        interaction_term = -0.5 * np.sum(self.J * np.outer(spins, spins))
        
        # 外部场项
        field_term = -np.sum(self.h * spins)
        
        # 传感器读数项
        sensor_term = self.lambda_reg * np.sum(d_sample**2 * spins)
        
        total_energy = interaction_term + field_term + sensor_term
        
        return total_energy
    
    def compute_prediction_probability(self, spins, d_sample):
        """改进的预测概率计算"""
        probabilities = np.zeros(self.n_sensors)
        
        for i in range(self.n_sensors):
            # 计算每个传感器的能量差
            interaction_sum = np.sum(self.J[i] * spins)
            
            # 计算翻转能量差
            delta_E = 2 * spins[i] * (interaction_sum + self.h[i] - self.lambda_reg * d_sample[i]**2)
            
            # 使用Boltzmann分布计算概率
            # 故障概率 = 1 / (1 + exp(ΔE/T))
            fault_prob = 1.0 / (1.0 + np.exp(delta_E / self.temperature))
            probabilities[i] = fault_prob
        
        return probabilities
    
    def compute_cross_entropy_loss(self, y_true, y_pred_prob):
        """改进的交叉熵损失计算"""
        # 将标签转换为二进制
        y_binary = 1.0 if y_true != 1 else 0.0
        
        # 计算样本级别的平均故障概率
        sample_fault_prob = np.mean(y_pred_prob)
        
        # 避免log(0)
        epsilon = 1e-15
        sample_fault_prob = np.clip(sample_fault_prob, epsilon, 1 - epsilon)
        
        # 计算二元交叉熵
        ce_loss = -(y_binary * np.log(sample_fault_prob) + 
                   (1 - y_binary) * np.log(1 - sample_fault_prob))
        
        return ce_loss
    
    def update_temperature(self, iteration):
        """温度退火调度"""
        # 线性退火
        progress = iteration / self.max_iter
        self.temperature = self.initial_temp - progress * (self.initial_temp - self.final_temp)
        return self.temperature
    
    def metropolis_update(self, spins, d_sample):
        """改进的蒙特卡洛更新"""
        new_spins = spins.copy()
        accepted_flips = 0
        
        for step in range(self.mc_steps):
            # 随机选择一个节点
            i = np.random.randint(self.n_sensors)
            
            # 计算当前能量
            current_energy = self.compute_energy(new_spins, d_sample)
            
            # 翻转选定的自旋
            new_spins[i] = -new_spins[i]
            
            # 计算新能量
            new_energy = self.compute_energy(new_spins, d_sample)
            
            # 计算能量差
            delta_E = new_energy - current_energy
            
            # Metropolis接受准则
            if delta_E <= 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(-delta_E / self.temperature)
            
            # 决定是否接受翻转
            if np.random.random() < accept_prob:
                accepted_flips += 1
            else:
                # 拒绝翻转，恢复原状态
                new_spins[i] = -new_spins[i]
        
        acceptance_rate = accepted_flips / self.mc_steps
        return new_spins, acceptance_rate
    
    def fit(self, X, y):
        """训练改进版伊辛模型"""
        print("开始训练改进版伊辛模型...")
        print(f"训练参数: alpha={self.alpha}, beta={self.beta}, lambda={self.lambda_reg}")
        print(f"温度: {self.initial_temp} → {self.final_temp}, 最大迭代={self.max_iter}")
        
        # 数据预处理
        X_processed, d_values = self.preprocess_data(X, y)
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            iter_start = time.time()
            
            # 更新温度
            current_temp = self.update_temperature(iteration)
            
            # 随机选择一个训练样本
            sample_idx = np.random.randint(len(X_processed))
            d_sample = d_values[sample_idx]
            y_sample = y[sample_idx]
            
            # 1. 计算当前能量
            energy = self.compute_energy(self.spins, d_sample)
            
            # 2. 计算预测概率
            y_pred_prob = self.compute_prediction_probability(self.spins, d_sample)
            
            # 3. 计算交叉熵损失
            ce_loss = self.compute_cross_entropy_loss([y_sample], y_pred_prob)
            
            # 4. 计算总损失
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * energy
            
            # 5. 蒙特卡洛状态更新
            self.spins, acceptance_rate = self.metropolis_update(self.spins, d_sample)
            
            # 6. 计算精度
            fault_pred = (y_pred_prob >= 0.5).astype(int)
            y_binary = 1 if y_sample != 1 else 0
            sample_accuracy = 1 if (np.any(fault_pred) == y_binary) else 0
            
            # 记录历史
            self.loss_history.append(total_loss)
            self.energy_history.append(energy)
            self.ce_loss_history.append(ce_loss)
            self.accuracy_history.append(sample_accuracy)
            self.temperature_history.append(current_temp)
            
            iter_time = time.time() - iter_start
            
            # 打印进度
            if iteration % 20 == 0 or iteration == self.max_iter - 1:
                avg_accuracy = np.mean(self.accuracy_history[-20:]) if len(self.accuracy_history) >= 20 else sample_accuracy
                print(f"Iter {iteration:3d}: Loss={total_loss:.4f} (CE={ce_loss:.4f}, E={energy:.4f}), "
                      f"Acc={avg_accuracy:.3f}, Temp={current_temp:.3f}, "
                      f"AcceptRate={acceptance_rate:.3f}, Spins(+1)={np.sum(self.spins==1)}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
        print(f"最终参数:")
        print(f"  - 温度: {self.temperature:.3f}")
        print(f"  - J矩阵: mean={np.mean(self.J):.4f}, std={np.std(self.J):.4f}")
        print(f"  - h向量: mean={np.mean(self.h):.4f}, std={np.std(self.h):.4f}")
        print(f"  - 自旋状态: {np.sum(self.spins==1)} 正常, {np.sum(self.spins==-1)} 故障")
        
    def predict(self, X):
        """预测测试集"""
        print("开始预测...")
        
        # 数据预处理
        X_processed, d_values = self.preprocess_data(X)
        
        predictions = []
        probabilities = []
        
        for i in range(len(X_processed)):
            # 计算每个样本的预测概率
            y_pred_prob = self.compute_prediction_probability(self.spins, d_values[i])
            
            # 样本级别的故障概率
            sample_fault_prob = np.mean(y_pred_prob)
            
            # 根据阈值判断故障
            is_fault = 1 if sample_fault_prob >= 0.5 else 0
            
            predictions.append(2 if is_fault else 1)  # 2=故障, 1=正常
            probabilities.append(sample_fault_prob)
        
        print(f"预测完成，共 {len(predictions)} 个样本")
        print(f"预测概率范围: [{min(probabilities):.3f}, {max(probabilities):.3f}]")
        
        return np.array(predictions), np.array(probabilities)
    
    def plot_training_history(self, save_path="improved_training_history.png"):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 总损失
        axes[0, 0].plot(self.loss_history)
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # 能量和温度
        axes[0, 1].plot(self.energy_history, label='Energy')
        ax2 = axes[0, 1].twinx()
        ax2.plot(self.temperature_history, 'r-', label='Temperature')
        axes[0, 1].set_title('Energy and Temperature')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Energy')
        ax2.set_ylabel('Temperature')
        axes[0, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[0, 1].grid(True)
        
        # 精度
        window_size = 50
        moving_avg = np.convolve(self.accuracy_history, np.ones(window_size)/window_size, mode='valid')
        axes[1, 0].plot(moving_avg)
        axes[1, 0].set_title(f'Moving Average Accuracy (window={window_size})')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True)
        
        # 自旋状态变化
        spin_counts = [np.sum(self.spins == 1) for _ in range(len(self.loss_history))]
        axes[1, 1].plot(spin_counts)
        axes[1, 1].set_title('Number of Normal Spins')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"训练历史图已保存: {save_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("改进版伊辛模型故障检测")
    print("=" * 60)
    
    # 1. 加载数据
    data_file = "Datasets/data_1ohm_50db.csv"
    print(f"加载数据: {data_file}")
    
    data = pd.read_csv(data_file, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 2. 数据划分
    print("\n数据划分...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 3. 创建和训练改进版模型
    print("\n创建改进版伊辛模型...")
    model = ImprovedIsingFaultDetector(
        n_sensors=354,
        alpha=0.6,
        beta=0.005,
        lambda_reg=0.2,
        initial_temp=2.0,
        final_temp=0.5,
        max_iter=200,
        mc_steps=50
    )
    
    # 训练模型
    print("\n" + "="*40)
    model.fit(X_train, y_train)
    
    # 4. 预测和评估
    print("\n" + "="*40)
    y_pred, y_prob = model.predict(X_test)
    
    # 将多分类问题简化为二分类（正常 vs 故障）
    y_test_binary = (y_test != 1).astype(int)
    y_pred_binary = (y_pred != 1).astype(int)
    
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    print(f"\n模型评估:")
    print(f"  - 准确率: {accuracy:.4f}")
    print(f"  - 预测概率统计: mean={np.mean(y_prob):.3f}, std={np.std(y_prob):.3f}")
    
    # 详细分类报告
    print(f"\n分类报告:")
    print(classification_report(y_test_binary, y_pred_binary, 
                              target_names=['Normal', 'Fault']))
    
    # 5. 绘制训练历史
    model.plot_training_history("improved_ising_training_history.png")
    
    # 6. 分析预测概率分布
    plt.figure(figsize=(10, 4))
    plt.hist(y_prob, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Predicted Fault Probability Distribution')
    plt.xlabel('Fault Probability')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("概率分布图已保存: probability_distribution.png")
    
    print("\n" + "="*60)
    print("改进版伊辛模型故障检测完成!")
    print("="*60)


if __name__ == "__main__":
    main()