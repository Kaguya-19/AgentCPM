

<div align="center">
  <img src="./assets/light.svg" alt="AgentCPM-Explore 标志" width="400em"></img>
</div>

<p align="center">
    【中文 | <a href="README.md">English</a>】
</p>



# 最新消息

* [2026-01-12] 🚀🚀🚀我们开源了基于全量仅**4B参数**的智能体大模型AgentCPM-Explore及其所有训练、推理、工具沙盒环境代码，成功闯入GAIA、HLE、BrowseComp等8个经典长难智能体任务榜单，同级别SOTA的表现带来更长行为链路、更准确的深度调研能力，由此突破端侧智能体的性能壁垒。


# 概述
AgentCPM 是由[清华大学自然语言处理实验室（THUNLP）](https://nlp.csai.tsinghua.edu.cn)、[中国人民大学](http://ai.ruc.edu.cn/)、[面壁智能](https://modelbest.cn/en)以及[OpenBMB社区](https://www.openbmb.cn/home)联合开发的一系列开源大语言模型智能体。针对智能体在真实世界应用时所面临的长程性、自主性、泛化性不足的问题，提出一系列模型构建方案。团队近期聚焦于先对智能体的深度研究能力进行全方位构建，发布[AgentCPM-Explore](./AgentCPM-Explore/README_zh.md)深度搜索大语言模型智能体与AgentCPM-Report深度调研大语言模型智能体。

# 模型列表

| 模型            | 下载链接                                                                                                                                | 开源内容 | 技术报告 |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------|
| [AgentCPM-Explore](./AgentCPM-Explore)          | [🤗 Hugging Face](https://huggingface.co/openbmb/AgentCPM-Explore)<br> [🤖 ModelScope](https://modelscope.cn/models/OpenBMB/AgentCPM-Explore/)                  |  [AgentDock](./AgentCPM-Explore/AgentDock/README_zh.md): 工具沙盒环境统一管理调度平台  <br> [AgentRL](./AgentCPM-Explore/AgentRL/README_zh.md): 全异步智能体强化学习框架  <br> [AgentToLeaP](./AgentCPM-Explore/AgentToLeaP/README_zh.md): 智能体工具学习能力一键测评框架 | 即将发布 |


## AgentCPM-Explore

### 简介
**AgentCPM-Explore** 拥有 40 亿参数，取得同尺寸模型SOTA、越级赶上甚至超越两倍大参数量（8B级）SOTA模型、比肩部分30B级以上和闭源大模型的效果，实现了更长的行为链路和更准确的深度调研（Deep Research）能力，真正让大模型的长程任务处理能力有望部署于端侧，加速私有化智能助手的普及。AgentCPM-Explore的亮点包括：

- 首个以 4B 全量参数登上 GAIA、HLE、BrowseComp 等 8 个长程复杂智能体任务榜单的端侧智能体模型。

- 可实现超过 100 轮的连续环境交互，支持多源信息交叉验证、搜索策略动态调整、实时核验最新信息，持续深度探索直至任务完成。

- 全流程开源，包括智能体全异步强化学习训练框架AgentRL、工具沙盒统一管理调度平台AgentDock、智能体工具学习能力一键测评平台AgentToLeaP，支持社区共建与自定义扩展。


### 持续深度探索
演示案例（倍速）：


https://github.com/user-attachments/assets/f8487889-d17a-447e-9aef-2608f4c84a83


# 开源协议

* 本仓库开源的代码遵照 [Apache-2.0](./LICENSE) 协议。


# 引用
如果 **AgentCPM-Explore** 对您的研究有帮助，您可以按照如下方式进行引用

```bibtex
@software{AgentCPMExplore2026,
  title  = {AgentCPM-Explore: An End-to-End Infrastructure for Training and Evaluating LLM Agents},
  author = {Haotian Chen and Xin Cong and Shengda Fan and Yuyang Fu and Ziqin Gong and Yaxi Lu and Yishan Li and Boye Niu and Chengjun Pan and Zijun Song and Huadong Wang and Yesai Wu and Yueying Wu and Zihao Xie and Yukun Yan and Zhong Zhang and Yankai Lin and Zhiyuan Liu and Maosong Sun},
  year   = {2026},
  url    = {https://github.com/OpenBMB/AgentCPM}
}
```



# 更多项目

- [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)
- [MiniCPM](https://github.com/OpenBMB/MiniCPM)
- [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)
- [UltraRAG](https://github.com/OpenBMB/UltraRAG)



