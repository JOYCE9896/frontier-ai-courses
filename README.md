# 前沿 AI 名校课

Stanford、CMU、Harvard 和 MIT 2025-2026 学年 AI 课程的中文笔记，附一套给零基础读者的入门教程。

**[在线阅读 →](https://joyce9896.github.io/frontier-ai-courses/)**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-204e79?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## 适合谁

- 想系统了解大模型、Agent 和 AI 产业，但不方便看英文原课的读者
- 没有机器学习基础：先读 [AI 基础](https://joyce9896.github.io/frontier-ai-courses/00-Foundations/)，15 篇，会中学数学就能看懂
- 已经入门：直接按下面的路线读课程笔记

## 怎么读

**零基础** → AI 基础 → CS224N → 11-711 → CS336 → 11-664 → CS329A / 11-768 / CS2680 / CS146S

非技术类课程（MS&E 435、CS153）不需要数学基础，可以直接读。想补计算机系统基础，读 CMU 15-213，会一点 C 就能开始；想知道程序怎样被编译成机器代码，接着读 Stanford CS143；想知道多核 CPU 和 GPU 怎样并行、为什么 FlashAttention 快，读 Stanford CS149；想知道多台机器怎样在故障中保持一致、Raft 怎样工作，读 MIT 6.5840。

## 课程

### 入门

| | 内容 |
| :-- | :-- |
| [AI 基础](https://joyce9896.github.io/frontier-ai-courses/00-Foundations/) | 向量、概率、导数、神经网络、Transformer、训练与解码，每篇用具体数字算一遍 |
| [动手练习](https://joyce9896.github.io/frontier-ai-courses/01-Labs/) | 能跑的小程序，代码和实际运行结果原样贴出：从零训练一个小 LLaMA 等 |

### CS 基础

| 课程 | 内容 |
| :-- | :-- |
| [CMU 15-213 (CS:APP)](https://joyce9896.github.io/frontier-ai-courses/15-213-CSAPP/) | 计算机系统入门，CS:APP 两位作者亲授的 2015 年秋季版全部 27 讲：数据表示、x86-64 机器代码、高速缓存、链接、进程与信号、虚拟内存、malloc、网络、并发 |
| [Stanford CS143](https://joyce9896.github.io/frontier-ai-courses/CS143-Compilers/) | 编译器：2026 年春季课件加 Aiken 公开课，18 讲覆盖词法分析、LL 和 LR 语法分析、类型检查、代码生成、操作语义、数据流分析、寄存器分配、垃圾回收，附 Cool 编译器项目和书面作业梳理 |
| [Stanford CS149](https://joyce9896.github.io/frontier-ai-courses/CS149-Parallel-Computing/) | 并行计算：2023 年秋季全部 19 讲，多核、SIMD、ISPC、CUDA、局部性与调度、Spark、DNN 与 FlashAttention、缓存与内存一致性、无锁编程、事务内存、DSL、专用硬件与 DRAM，附五个编程作业和书面作业梳理 |
| [MIT 6.5840](https://joyce9896.github.io/frontier-ai-courses/MIT-6.5840-Distributed-Systems/) | 分布式系统（原 6.824）：2026 年春季讲义对照 2020 年录像，21 讲覆盖 MapReduce、GFS、Paxos、Raft、线性一致性、ZooKeeper、两阶段提交、Spanner、链式复制、FaRM、Memcache、Lambda、Ray、SUNDR、比特币、拜占庭容错，另有七篇 2020 年版补充，附五个 Go 实验梳理和 2025 年两套考试的逐题解析 |

### 技术

| 课程 | 内容 |
| :-- | :-- |
| [Stanford CS224N](https://joyce9896.github.io/frontier-ai-courses/CS224N-NLP-with-Deep-Learning/) | NLP 与深度学习：从词向量到推理训练 |
| [CMU 11-711](https://joyce9896.github.io/frontier-ai-courses/11-711-Advanced-NLP/) | 高级 NLP：预训练、微调、检索、强化学习、量化与并行 |
| [Stanford CS336](https://joyce9896.github.io/frontier-ai-courses/CS336-Language-Modeling-from-Scratch/) | 从零实现语言模型 |
| [CMU 11-664/763](https://joyce9896.github.io/frontier-ai-courses/11-664-LM-Inference/) | 语言模型推理算法与推理效率 |
| [Stanford CS329A](https://joyce9896.github.io/frontier-ai-courses/CS329A-Self-Improving-AI-Agents/) | 自我改进的 Agent |
| [CMU 11-768](https://joyce9896.github.io/frontier-ai-courses/11-768-AI-Agents/) | AI Agent 的构建、训练与安全（课程进行中） |
| [Harvard CS2680](https://joyce9896.github.io/frontier-ai-courses/CS2680-Modern-AI-Systems/) | 现代 AI 系统：从 Claude Code、Agent 设计到 LLM 推理服务、GPU 编程、批处理调度与并行，附五个作业梳理（课程进行中） |
| [Stanford CS146S](https://joyce9896.github.io/frontier-ai-courses/CS146S-Modern-Software-Developer/) | 现代软件开发者：编程 Agent 的内部、RePPIT 规格驱动开发、MCP、Skill、常驻 Agent、AGENTS.md、Agent 就绪的代码库、Agent 代码审查、后台 Agent，附五个作业梳理，另有 11 篇讲者访谈补充（课程进行中） |
| [Stanford CS25](https://joyce9896.github.io/frontier-ai-courses/CS25-Transformers-United/) | Transformer 前沿研讨 |
| [Stanford CME296](https://joyce9896.github.io/frontier-ai-courses/CME296-Diffusion-and-Large-Vision-Models/) | 扩散模型与视觉生成 |
| [Andrej Karpathy YouTube](https://joyce9896.github.io/frontier-ai-courses/Andrej-Karpathy-YouTube/) | Karpathy 全部 17 条视频，每条一篇：micrograd、makemore、从零构建 GPT、分词器、复现 GPT-2、LLM 原理与日常用法，外加一篇讲法拆解 |
| [吴恩达 Agentic AI](https://joyce9896.github.io/frontier-ai-courses/Andrew-Ng-Agentic-AI/) | DeepLearning.AI 的 Agent 构建课，5 个模块 31 节视频：反思、工具使用与 MCP、评测与错误分析、规划、多 Agent 协作 |
| [AI Engineer 大会直播](https://joyce9896.github.io/frontier-ai-courses/AI-Engineer-Livestreams/) | AI Engineer 大会 YouTube 全部 38 条直播（2023–2026），37 篇、670 节演讲笔记，每节附直播时间点链接：RAG、Agent、MCP、评测、编程 Agent、Harness 工程 |
| [Dwarkesh Podcast](https://joyce9896.github.io/frontier-ai-courses/Dwarkesh-Podcast/) | 2025-09 到 2026-11 最好的 21 期 AI 访谈加一篇短文合集，22 篇、211 节：Karpathy、Ilya、Sutton、Dario、Satya、黄仁勋、Musk、Dylan Patel、Reiner Pope、递归自我改进讨论 |
| [Lauren Tan](https://joyce9896.github.io/frontier-ai-courses/Lauren-Tan-Grok-Bot/) | Grok Bot 工程负责人“土豆姐”2026 年 9–11 月的 3 期访谈和演讲，34 节：团队在用的 Bot、一个月 11,000 个 PR 怎样守住质量、AI 同事的身份和权限 |
| [How to Scale Your Model](https://joyce9896.github.io/frontier-ai-courses/How-to-Scale-Your-Model/) | Google DeepMind 的大模型扩展书，导读加 12 章：roofline、TPU/GPU、分片与集合通信、训练并行、推理、LLaMA 3 算例，含习题解答 |
| [Smol Training Playbook](https://joyce9896.github.io/frontier-ai-courses/Smol-Training-Playbook/) | Hugging Face 训练 SmolLM3 的完整记录，导读加 8 章：消融实验、架构与超参数、数据配方、训练中排错、SFT/偏好优化/RL 后训练、GPU 集群基础设施 |
| [Ultra-Scale Playbook](https://joyce9896.github.io/frontier-ai-courses/Ultra-Scale-Playbook/) | Hugging Face 的 GPU 集群训练手册，导读加 12 篇：显存账、数据并行与 ZeRO、张量/序列/上下文/流水线/专家并行、5D 组合与选配置、GPU kernel、FlashAttention、FP8 混合精度 |
| [Lilian Weng 博客](https://joyce9896.github.io/frontier-ai-courses/Lilian-Weng-Blog/) | Lil'Log 15 篇长综述：2024 年以来全部 7 篇（harness 工程、scaling law、测试时思考、奖励作弊、幻觉、视频扩散、人类数据）加 agent、prompt、推理优化等 8 篇经典 |
| [ETH 263-5911 Robot Learning](https://joyce9896.github.io/frontier-ai-courses/ETH-263-5911-Robot-Learning/) | 机器人学习，2026 年春季：模仿学习、强化学习、扩散策略、动作 tokenization、世界模型、VLA（π0、OpenVLA）与具身推理，11 讲加 10 场嘉宾讲座 |
| [Stanford CS234](https://joyce9896.github.io/frontier-ai-courses/CS234-Reinforcement-Learning/) | 强化学习，2024 年春季录像 16 讲：MDP、TD、Q-learning、策略梯度、PPO、RLHF 与 DPO、离线 RL、老虎机与探索、MCTS 与 AlphaZero，附 2026 年冬季课件更新 |

### 产业与系统

| 课程 | 内容 |
| :-- | :-- |
| [Stanford MS&E 435](https://joyce9896.github.io/frontier-ai-courses/MSE435-Economics-of-the-AI-Supercycle/) | AI 产业经济：钱流向哪里 |
| [Stanford CS153](https://joyce9896.github.io/frontier-ai-courses/CS153-Frontier-Systems/) | 前沿系统：能源、算力、模型公司与创业 |
| [BG2 Pod](https://joyce9896.github.io/frontier-ai-courses/BG2-Pod/) | 投资人眼里的 AI：黄仁勋两次长谈、Altman 与 Nadella、ChatGPT、企业 AI、AI 资本开支，7 期 109 节 |

## 资料来源

笔记依据各课程公开的课堂录像和幻灯片整理，每篇开头列出依据的资料；没有公开资料的讲次会单独注明。目前正在逐门与原始录像核对，发现问题欢迎提 Issue。

## 许可

[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)：可以转载和修改，需署名、非商业使用、以相同方式共享。课程内容的著作权归原学校、授课教师和嘉宾所有。
