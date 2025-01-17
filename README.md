# TiSASRec_ReChorus_@course ML2024

## 项目简介

本项目是对中山大学机器学习课程作业中提到的 TiSASRec 模型的复现。TiSASRec（Time Interval Aware Self-Attention for Sequential Recommendation）是一种用于顺序推荐的时间间隔感知自注意力模型，旨在通过融合项目位置与时间间隔信息，提升推荐系统的准确性和个性化体验。

## 项目结构

TiSASRec_ReChorus/
├── data/
│   ├── Groceries/
│   ├── MIND-Large/
│   └── MovieLens/
├── models/
│   └── TiSASRec.py
├── utils/
│   ├── preprocessing.py
│   ├── reader.py
│   └── evaluation.py
├── train.py
├── test.py
└── README.md

## 数据集

本项目使用了以下三个数据集进行实验：

- **Groceries**：聚焦食品杂货，数据稀疏，用户购买低频。
- **MIND-Large**：具有独特特征，在数据分布和用户行为上与其他两者有别。
- **MovieLens**：数据丰富且密集，用户行为多样频繁。

## 模型架构

TiSASRec 模型的核心在于其时间间隔感知自注意力机制，该机制能够同时考虑项目的时间间隔和位置信息，通过创新的加权和计算方式及兼容性函数确定权重系数，实现对用户行为序列的精准建模。

## 实验结果

在多个数据集上的实验结果显示，TiSASRec 模型在 NDCG 和 HR 指标上均优于其他基线方法，充分彰显了模型在捕捉用户偏好、融合多维度信息及提升推荐准确性方面的优势。

## 使用方法

### 环境依赖

- Python 3.6+
- PyTorch 1.4+
- NumPy
- Pandas

### 安装依赖

```bash
pip install -r requirements.txt
```

#### 数据预处理

```bash
python utils/preprocessing.py
```

#### 训练模型

```bash
python train.py
```

#### 测试模型

```bash
python test.py
```

#### 测试结果

![1736948758398](images/README/1736948758398.png)

## 项目仓库

[https://github.com/liuyh357/TiSASRec_ReChorus.git](https://github.com/liuyh357/TiSASRec_ReChorus.git)

## 主要参考文献

- Li, Jiacheng, Yujie Wang, and Julian McAuley (2020). “Time Interval Aware Self-Attention for Sequential Recommendation”. In: Proceedings of the 13th International Conference on Web Search and Data Mining. WSDM ’20. Houston, TX, USA: Association for Computing Machinery, pp. 322–330. ISBN: 9781450368223. DOI: [10.1145/3336191.3371786](https://doi.org/10.1145/3336191.3371786).
- Li, Jiayu et al. (2024). ReChorus2.0: A Modular and Task-Flexible Recommendation Library. arXiv: [2405.18058 [cs.IR]](https://arxiv.org/abs/2405.18058).

---
