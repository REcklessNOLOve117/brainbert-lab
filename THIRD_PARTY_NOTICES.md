# 来源与第三方说明

本页面的架构讲解依据 czlwang/brainbert 的 large 配置与相关实现，原始仓库：https://github.com/czlwang/brainbert ，参考版本 cd2aaa8。

数据文件 docs/brain-data.json 中的 waveform 来源为该仓库 notebooks/example_wav_1.npy；STFT 数值由本项目脚本计算导出。采样率与电压单位说明依据原 demo；npy 本身没有单位元数据。

页面展示了原实现的短代码节选并附源码链接。真实信号与原始实现的权利及使用条件以原始发布方为准。本项目不重新授权第三方材料。

本项目不包含预训练模型权重。

PopT 章节的架构与公式说明依据 czlwang/PopulationTransformer（https://github.com/czlwang/PopulationTransformer），参考版本 d237755bbdea123c60bface0cecbb48bab6f42de；论文 https://arxiv.org/abs/2406.03044 。本项目对架构与数学计算作独立可视化实现；多电极坐标和活动均为合成教学输入，没有复制真实受试者的多电极数据。官方权重入口为 https://huggingface.co/PopulationTransformer/popt_brainbert_stft ，本项目未分发或加载该权重。
