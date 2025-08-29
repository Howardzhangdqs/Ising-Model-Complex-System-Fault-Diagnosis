# Ising Model for Complex System Fault Diagnosis

## 基于伊辛模型的复杂系统故障诊断方法

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org)
[![Research](https://img.shields.io/badge/Type-Academic%20Research-blue.svg)]()
[![Status](https://img.shields.io/badge/Status-Research%20Complete-green.svg)]()

### Abstract | 摘要

This repository contains the implementation and evaluation of a novel fault localization method for complex systems based on the Ising model from statistical physics. The method maps sensors in complex systems to nodes in an Ising lattice and utilizes energy minimization principles to identify faulty components. The approach has been validated on the IEEE 118-bus power system dataset.

本研究提出了一种基于统计物理学伊辛模型的复杂系统故障定位新方法。该方法将复杂系统中的传感器映射为伊辛晶格中的节点，利用能量最小化原理识别故障组件。方法在IEEE 118母线电力系统数据集上得到验证。

---

## 🎯 Research Objectives | 研究目标

- **理论创新**: 将伊辛模型引入复杂系统故障诊断领域
- **方法验证**: 在标准电力系统数据集上验证方法有效性  
- **工程应用**: 为复杂系统提供可解释的故障定位方案
- **性能评估**: 全面分析方法的准确性、效率和适用性

---

## 📊 Dataset | 数据集

**IEEE 118-Bus Power System Dataset**
- **Source**: Cyber-Physical Power System simulation data
- **Samples**: 17,500 (250 samples × 70 scenarios)
- **Features**: 354 (118 voltage + 118 frequency + 118 phase angle)
- **Scenarios**: 70 operational states (1 normal + 69 fault types)
- **Fault Types**: Load Loss (LL), Generator Outage (GO), Generator Ground (GG)

---

## 🔬 Methodology | 方法论

### Core Algorithm | 核心算法

1. **数据预处理**: 传感器数据标准化和特征工程
2. **伊辛映射**: 传感器状态映射为自旋变量
3. **能量函数**: 定义包含相互作用项、外场项和传感器读数项的能量函数
4. **参数优化**: 使用梯度下降最小化损失函数
5. **蒙特卡洛更新**: 采用Metropolis准则更新节点状态
6. **故障判定**: 基于最终自旋状态确定故障位置

### Energy Function | 能量函数

```
E = -∑(i,j) J_ij * s_i * s_j - ∑_i h_i * s_i + λ∑_i d_i² * s_i
```

其中：
- `J_ij`: 相互作用参数
- `s_i`: 节点自旋状态 (+1正常, -1故障)
- `h_i`: 外部场参数
- `d_i`: 传感器读数偏差
- `λ`: 正则化参数

---

## �️ Implementation | 实现版本

### Core Files | 核心文件

1. **`ising_fault_detection.py`** - 基础实现版本
   - 完整伊辛模型框架
   - 基本蒙特卡洛算法
   - **性能**: 准确率 98.57%

2. **`ising_fault_detection_v2.py`** - 改进实现版本  
   - 添加温度退火机制
   - 优化参数更新策略
   - **性能**: 准确率 20.19% (调试中)

3. **`ising_fault_detection_final.py`** - 实用化版本
   - 简化但稳定的实现
   - 针对实际应用优化
   - **性能**: 准确率 50.90%

### Documentation | 文档

- **`model.md`** - 完整技术规范和权利要求书
- **`model_analysis_report.md`** - 模型分析报告
- **`final_evaluation_report.md`** - 综合评估报告
- **`数据集格式说明文档.md`** - 数据集详细说明

---

## 📈 Results | 实验结果

### Performance Summary | 性能总结

| 版本 | 准确率 | 训练时间 | 特点 |
|------|--------|----------|------|
| 基础版本 | 98.57% | <10s | 高准确率，快速收敛 |
| 改进版本 | 20.19% | ~15s | 温度退火，概率分布优化 |
| 实用版本 | 50.90% | <5s | 稳定可靠，计算高效 |

### Key Findings | 主要发现

✅ **技术可行性**: 伊辛模型在电力系统故障检测中表现出良好潜力  
✅ **计算效率**: 训练时间短，适合实时应用  
✅ **物理可解释性**: 提供传感器级别的故障定位  
⚠️ **改进空间**: 正常样本识别和多分类精度需要优化

---

## 🚀 Usage | 使用方法

### Quick Start | 快速开始

```bash
# 1. 准备数据集
mkdir -p Datasets
# 下载IEEE 118-Bus数据集到 Datasets/ 目录

# 2. 运行基础版本
python ising_fault_detection.py

# 3. 运行实用版本  
python ising_fault_detection_final.py

# 4. 查看结果
ls *.png  # 查看生成的可视化图表
```

### Dependencies | 依赖包

---

## 📋 Project Structure | 项目结构

```
├── ising_fault_detection.py          # 基础实现
├── ising_fault_detection_v2.py       # 改进版本
├── ising_fault_detection_final.py    # 实用版本
├── model.md                          # 技术规范
├── model_analysis_report.md          # 模型分析
├── final_evaluation_report.md        # 综合评估
├── 数据集格式说明文档.md              # 数据集说明
├── Datasets/                         # 数据集目录
├── *.png                            # 可视化结果
└── README.md                        # 本文档
```

---

## 🔮 Future Work | 未来工作

### Short-term | 短期改进
- [ ] 优化正常样本识别能力
- [ ] 系统化超参数调优
- [ ] 实现完整的70类分类
- [ ] 添加异常检测机制

### Long-term | 长期发展  
- [ ] 在线学习和实时更新
- [ ] 分布式计算支持
- [ ] 硬件加速实现
- [ ] 扩展到其他复杂系统

---

## 📖 Publications | 相关发表

*待发表的学术论文和会议报告将在此更新*

---

## 👥 Contributors | 贡献者

**主要研究者**: [您的姓名]  
**研究机构**: [您的机构]  
**联系方式**: [您的邮箱]  

---

## 📄 License | 许可证

This research project is for academic purposes. Please cite this work if you use it in your research.

本研究项目仅供学术使用。如在研究中使用，请引用本工作。

---

## 📚 Citation | 引用格式

```bibtex
@misc{ising_fault_diagnosis_2025,
  title={A Novel Fault Localization Method for Complex Systems Based on Ising Model},
  author={[Your Name]},
  year={2025},
  institution={[Your Institution]},
  note={Available at: https://github.com/[username]/Ising-Model-Complex-System-Fault-Diagnosis}
}
```

---


