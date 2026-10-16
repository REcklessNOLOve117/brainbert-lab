# BrainBERT + PopT Interactive Lab

一个中文交互式模型学习页面，从真实的单电极信号出发，理解 BrainBERT 的模型架构与表示学习，并进一步认识 Population Transformer（PopT）的多电极聚合。

## 八个章节

1. 模型架构总览：输入编码、六层 Encoder、MHA、残差、LayerNorm、FFN，以及三个任务出口。
2. 原始波形与 STFT：滑动窗口、40 个频率特征，幅值与 z-score 切换。
3. 时间 token 与 40→768 输入投影、正弦位置编码。
4. 多头注意力：12 个 head、Q/K/V、softmax、padding。
5. 遮挡、模拟重建与实际 L1 计算公式。
6. embedding、重建输出和分类输出。
7. 冻结主干、全量微调和 LoRA 扩展教学方案。
8. PopT 多电极模型：每电极时间池化、L/I/P/组别位置编码、CLS、跨电极注意力、两个 BCE 自监督任务与下游分类。

## 计算与数据边界

真实样本来自 [czlwang/brainbert](https://github.com/czlwang/brainbert) 的 `notebooks/example_wav_1.npy`（参考版本 `cd2aaa8`）。STFT 采用训练预处理顺序：2048 Hz、Hann 400 点、步长 50、前 40 频点、逐频率 z-score 后两侧各裁 10 帧。

本页面未加载 `stft_large_pretrained.pth`。输入投影与注意力采用固定随机参数；重建使用目标加可控误差；分类 logit 手动设定。L1、BCE、softmax 等按实际公式计算，但演示数值不代表预训练模型性能。LoRA 为扩展教学说明，原仓库未实现。

PopT 章节依据 [czlwang/PopulationTransformer](https://github.com/czlwang/PopulationTransformer) 的 `d237755bbdea123c60bface0cecbb48bab6f42de`。PopT 指 Population Transformer，与 PEFT/LoRA 无关。本项目未加载 PopT 权重，多电极布局与特征是合成教学数据，注意力由固定随机参数计算，不可解读为真实脑区连接或模型性能。

## 从已登录的 Windows PowerShell 发布

项目文件夹中运行：

```powershell
.\Publish-BrainBERT.ps1
```

脚本默认使用 GitHub 账号 `REcklessNOLOve117`，新建公开仓库 `brainbert-lab`，推送代码后把 `main` 分支的 `/docs` 目录设为 GitHub Pages 来源。使用本机 `gh` 的现有登录态，不索取、复制或保存访问令牌。

同名仓库若并非脚本本次创建，脚本会停止，避免覆盖。可通过 `-RepoName` 指定其他新名字。

GitHub 初次构建可能需要几分钟；脚本会区分已部署、仍在构建和失败状态。

## 本地阅读

在项目目录启动任意静态 HTTP 服务，例如：

```bash
python -m http.server 8000 --directory docs
```

然后打开 `http://localhost:8000/`。这是纯静态页面，不需要 Node 依赖或模型权重。ES modules 与 fetch 需要 HTTP 服务，不建议直接双击 HTML 文件。

## 数学验证

使用支持 ES modules 的现代 Node.js 运行：

```bash
node --test tests/popt-math.test.mjs
```

检查位置编码分段、BCE 标签映射与数值稳定性、注意力归一化和加权汇总，以及交互输入对计算结果的影响。

## 重新导出真实样本

安装 NumPy、SciPy，并准备原始 BrainBERT 仓库：

```bash
python scripts/prepare_data.py --repo /path/to/brainbert --out docs/brain-data.json
```

## 来源

- [BrainBERT 论文](https://arxiv.org/abs/2302.14367)
- [原始仓库与预训练权重入口](https://github.com/czlwang/brainbert)
- [PopT 论文](https://arxiv.org/abs/2406.03044)
- [PopT 官方代码](https://github.com/czlwang/PopulationTransformer)
- [PopT 官方权重](https://huggingface.co/PopulationTransformer/popt_brainbert_stft)
- [数据与第三方说明](THIRD_PARTY_NOTICES.md)
