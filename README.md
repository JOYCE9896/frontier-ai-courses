# 前沿 AI 名校课（Stanford · CMU）

> 斯坦福大学（Stanford）九门、卡内基梅隆大学（CMU）三门 AI 前沿课程的个人研读笔记，2025-2026 学年，覆盖技术机制、系统工程、产业经济、公开研讨、法律治理、前沿系统栈、经典 NLP、语言模型推理、Agent 与视觉生成。

[![GitHub Pages](https://img.shields.io/badge/Reading-GitHub%20Pages-8b261e?style=flat-square&logo=github)](https://joyce9896.github.io/frontier-ai-courses/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-204e79?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**[→ 在线阅读全部笔记](https://joyce9896.github.io/frontier-ai-courses/)**

---

## 目录

- [课程列表](#课程列表)
- [排版原则](#排版原则)
- [本地查阅](#本地查阅)
- [声明](#声明)

---

## 课程列表

每门课的完整逐讲笔记只在对应课程主页里展开，这里只做索引。点课程名进对应主页看目录。

### 从零开始

| 内容 | 说明 |
| :--- | :--- |
| [AI 基础：从零开始](https://joyce9896.github.io/frontier-ai-courses/00-Foundations/) | 给零基础读者的前置知识，每篇讲一个概念，用具体数字算一遍再给公式，附各门课的推荐阅读顺序（陆续更新） |

### Technical

| 学校 | 课程 | 主讲 | 关注点 |
| :--- | :--- | :--- | :--- |
| Stanford | [CS329A: Self-Improving AI Agents](https://joyce9896.github.io/frontier-ai-courses/CS329A-Self-Improving-AI-Agents/) | Aakanksha Chowdhery、Azalia Mirhoseini | Agent 如何通过强化学习与环境反馈实现自我演进 |
| Stanford | [CS336: Language Modeling from Scratch](https://joyce9896.github.io/frontier-ai-courses/CS336-Language-Modeling-from-Scratch/) | Percy Liang、Tatsunori Hashimoto | 从字节到对齐模型，亲手实现完整语言模型工程栈 |
| Stanford | [CS25: Transformers United V6](https://joyce9896.github.io/frontier-ai-courses/CS25-Transformers-United/) | Steven Feng、Karan P. Singh、Christopher Manning | 斯坦福长年公开的 Transformers 前沿研讨课 |
| Stanford | [CS224N: NLP with Deep Learning](https://joyce9896.github.io/frontier-ai-courses/CS224N-NLP-with-Deep-Learning/) | Christopher Manning | 从词向量、Transformer 到推理训练，斯坦福最长寿的 NLP 课 |
| Stanford | [CME296: Diffusion & Large Vision Models](https://joyce9896.github.io/frontier-ai-courses/CME296-Diffusion-and-Large-Vision-Models/) | Afshine Amidi、Shervine Amidi | 扩散、分数匹配与流匹配的推导，潜空间、DiT 架构、训练蒸馏与评估 |
| CMU | [11-711: Advanced NLP](https://joyce9896.github.io/frontier-ai-courses/11-711-Advanced-NLP/) | Sean Welleck | 研究生 NLP 核心课（2026 春，23 讲）：语言模型、Transformer、预训练、微调与解码、检索、多模态、评估、强化学习、Agent、量化、并行、MoE、长序列与推理时扩展 |
| CMU | [11-664/763: Inference Algorithms for LMs](https://joyce9896.github.io/frontier-ai-courses/11-664-LM-Inference/) | Graham Neubig、Amanda Bertsch | 语言模型推理算法（2025 秋，已整理 15 讲）：采样与搜索、受控生成、思维链与推理模型、工具与 Agent、奖励模型、MBR、推理时扩展与效率 |
| CMU | [11-768: AI Agents](https://joyce9896.github.io/frontier-ai-courses/11-768-AI-Agents/) | Graham Neubig、Daniel Fried | 基于大语言模型的 Agent：工具、上下文、记忆、规划、SFT 与 RL 训练、RL 系统与安全（课程进行中，已整理 13 讲） |

### Non-Technical

| 学校 | 课程 | 主讲 | 关注点 |
| :--- | :--- | :--- | :--- |
| Stanford | [MS&E 435: Economics of the AI Supercycle](https://joyce9896.github.io/frontier-ai-courses/MSE435-Economics-of-the-AI-Supercycle/) | Apoorv Agrawal | 生成式 AI 的商业模式、基础设施资本开支与产业约束 |
| Stanford | [EE392B: Industrial AI](https://joyce9896.github.io/frontier-ai-courses/EE392B-Industrial-AI/) | Daniel O'Neill、Dimitry Gorinevsky | 制造业、军工、半导体等垂直行业的 AI 落地工程 |
| Stanford | [CS283: Governing Artificial Intelligence](https://joyce9896.github.io/frontier-ai-courses/CS283-Governing-AI/) | Nathaniel Persily、Rob Reich、Anka Reuel、Sanmi Koyejo | AI 治理的法律、政策与制度设计 |
| Stanford | [CS153: Frontier Systems](https://joyce9896.github.io/frontier-ai-courses/CS153-Frontier-Systems/) | Anjney Midha、Michael Abbott | 从能源、硅片拓扑、多模态仿真到千倍效能工程师的前沿系统全栈 |

---

## 排版原则

- **字体统一**：英文用 Source Serif 4，中文用思源宋体（Noto Serif SC），正文、代码块、表格共用一套字体，不混排无衬线体。
- **字号精简**：全站只有两级字号，标题 1.35rem，其余一律 1.05rem。
- **公式支持**：数学公式用 KaTeX 排印。

---

## 本地查阅

```bash
git clone https://github.com/JOYCE9896/frontier-ai-courses.git
cd frontier-ai-courses
open index.html
```

---

## 声明

本项目是个人学术研读整理，按 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 协议共享（署名、非商业性使用、相同方式共享）。课程大纲与讲座原始内容的著作权归斯坦福大学、卡内基梅隆大学、授课教师及演讲嘉宾所有。
