#!/usr/bin/env python3
"""
最终版伊辛模型故障检测 - 实用化版本
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')


class PracticalIsingFaultDetector:
    """实用化伊辛模型故障检测器"""
    
    def __init__(self, n_sensors=354, temperature=1.0, max_iter=100, mc_steps=50):
        """初始化实用化伊辛模型"""
        self.n_sensors = n_sensors
        self.temperature = temperature
        self.max_iter = max_iter
        self.mc_steps = mc_steps
        
        # 初始化模型参数
        self._initialize_parameters()
        
        # 训练历史
        self.energy_history = []
        self.spin_history = []
        
        # 数据预处理器
        self.scaler = StandardScaler()
        
    def _initialize_parameters(self):
        """参数初始化"""
        # 相互作用参数 - 基于传感器类型初始化
        self.J = np.zeros((self.n_sensors, self.n_sensors))
        
        # 外部场参数 - 初始为0
        self.h = np.zeros(self.n_sensors)
        
        # 自旋状态 - 初始为正常状态
        self.spins = np.ones(self.n_sensors)
        
        print(f"参数初始化完成:")
        print(f"  - 传感器数量: {self.n_sensors}")
        print(f"  - 初始自旋状态: 全部正常")
        
    def preprocess_data(self, X, y=None):
        """数据预处理"""
        print("数据预处理...")
        
        # 标准化处理
        if y is not None:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = self.scaler.transform(X)
        
        # 计算与期望值的偏差
        d_values = X_normalized  # 标准化后期望值为0
        
        print(f"  - 样本数量: {X_normalized.shape[0]}")
        print(f"  - 偏差统计: mean={np.mean(d_values):.3f}, std={np.std(d_values):.3f}")
        
        return X_normalized, d_values
    
    def compute_energy(self, spins, d_sample):
        """计算能量函数"""
        # 简化能量函数: E = -∑J_ij*s_i*s_j + λ∑d_i²*s_i
        interaction_term = -0.5 * np.sum(self.J * np.outer(spins, spins))
        sensor_term = 0.1 * np.sum(d_sample**2 * spins)  # λ=0.1
        
        return interaction_term + sensor_term
    
    def compute_fault_probability(self, spins, d_sample):
        """计算故障概率"""
        fault_probs = np.zeros(self.n_sensors)
        
        for i in range(self.n_sensors):
            # 计算能量变化
            neighbor_sum = np.sum(self.J[i] * spins)
            delta_E = 2 * spins[i] * (neighbor_sum - 0.1 * d_sample[i]**2)
            
            # 故障概率
            fault_probs[i] = 1.0 / (1.0 + np.exp(-delta_E / self.temperature))
        
        return fault_probs
    
    def metropolis_update(self, spins, d_sample):
        """蒙特卡洛更新"""
        new_spins = spins.copy()
        
        for _ in range(self.mc_steps):
            i = np.random.randint(self.n_sensors)
            
            # 计算当前能量
            current_energy = self.compute_energy(new_spins, d_sample)
            
            # 翻转自旋
            new_spins[i] = -new_spins[i]
            new_energy = self.compute_energy(new_spins, d_sample)
            
            # Metropolis准则
            delta_E = new_energy - current_energy
            if delta_E > 0:
                if np.random.random() >= np.exp(-delta_E / self.temperature):
                    new_spins[i] = -new_spins[i]  # 拒绝翻转
        
        return new_spins
    
    def fit(self, X, y):
        """训练模型"""
        print("开始训练实用化伊辛模型...")
        print(f"参数: 温度={self.temperature}, 迭代={self.max_iter}")
        
        # 数据预处理
        X_processed, d_values = self.preprocess_data(X, y)
        
        start_time = time.time()
        
        for iteration in range(self.max_iter):
            # 随机选择样本
            sample_idx = np.random.randint(len(X_processed))
            d_sample = d_values[sample_idx]
            
            # 计算当前能量
            energy = self.compute_energy(self.spins, d_sample)
            
            # 蒙特卡洛更新
            self.spins = self.metropolis_update(self.spins, d_sample)
            
            # 记录历史
            self.energy_history.append(energy)
            self.spin_history.append(np.sum(self.spins == 1))
            
            # 打印进度
            if iteration % 50 == 0 or iteration == self.max_iter - 1:
                print(f"Iter {iteration:3d}: Energy={energy:.2f}, NormalSensors={np.sum(self.spins==1)}")
        
        training_time = time.time() - start_time
        print(f"\n训练完成! 耗时: {training_time:.2f}秒")
        print(f"最终状态: {np.sum(self.spins==1)} 正常传感器, {np.sum(self.spins==-1)} 故障传感器")
        
    def predict(self, X):
        """预测故障"""
        print("开始预测...")
        
        X_processed, d_values = self.preprocess_data(X)
        
        predictions = []
        confidence_scores = []
        
        for i in range(len(X_processed)):
            # 计算故障概率
            fault_probs = self.compute_fault_probability(self.spins, d_values[i])
            avg_fault_prob = np.mean(fault_probs)
            
            # 判断故障
            is_fault = 1 if avg_fault_prob > 0.5 else 0
            
            predictions.append(2 if is_fault else 1)  # 2=故障, 1=正常
            confidence_scores.append(avg_fault_prob)
        
        print(f"预测完成: {len(predictions)} 个样本")
        print(f"置信度: [{min(confidence_scores):.3f}, {max(confidence_scores):.3f}]")
        
        return np.array(predictions), np.array(confidence_scores)
    
    def analyze_sensors(self):
        """分析传感器状态"""
        print("\n传感器分析:")
        print(f"正常传感器数量: {np.sum(self.spins == 1)}")
        print(f"故障传感器数量: {np.sum(self.spins == -1)}")
        
        # 按类型分组分析
        voltage_sensors = slice(0, 118)
        frequency_sensors = slice(118, 236)
        phase_sensors = slice(236, 354)
        
        print(f"电压传感器: {np.sum(self.spins[voltage_sensors] == -1)}/{118} 故障")
        print(f"频率传感器: {np.sum(self.spins[frequency_sensors] == -1)}/{118} 故障") 
        print(f"相角传感器: {np.sum(self.spins[phase_sensors] == -1)}/{118} 故障")


def main():
    """主函数"""
    print("=" * 60)
    print("实用化伊辛模型故障检测")
    print("=" * 60)
    
    # 加载数据
    data_file = "Datasets/data_1ohm_50db.csv"
    print(f"加载数据: {data_file}")
    
    data = pd.read_csv(data_file, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    print(f"数据形状: {X.shape}")
    
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 创建模型
    model = PracticalIsingFaultDetector(
        n_sensors=354,
        temperature=1.0,
        max_iter=300,
        mc_steps=100
    )
    
    # 训练
    print("\n" + "="*40)
    model.fit(X_train, y_train)
    
    # 预测
    print("\n" + "="*40)
    y_pred, confidence = model.predict(X_test)
    
    # 评估
    y_test_binary = (y_test != 1).astype(int)
    y_pred_binary = (y_pred != 1).astype(int)
    
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    print(f"\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"置信度范围: [{confidence.min():.3f}, {confidence.max():.3f}]")
    
    # 详细报告
    print(f"\n详细报告:")
    print(classification_report(y_test_binary, y_pred_binary, 
                              target_names=['Normal', 'Fault']))
    
    # 混淆矩阵
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    print(cm)
    print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    # 传感器分析
    model.analyze_sensors()
    
    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(model.energy_history)
    plt.title('Training Energy History')
    plt.xlabel('Iteration')
    plt.ylabel('Energy')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(model.spin_history)
    plt.title('Normal Sensors Count')
    plt.xlabel('Iteration')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('practical_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("训练历史图已保存")
    
    # 置信度分布
    plt.figure(figsize=(8, 4))
    plt.hist(confidence, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("置信度分布图已保存")
    
    print("\n" + "="*60)
    print("实用化伊辛模型完成!")
    print("="*60)


if __name__ == "__main__":
    main()