let architectureInstances = 0;

/** Mount the architecture lesson, returning a function that removes its listeners and DOM. */
export function mountArchitecture(container) {
  const uid = `ar-${++architectureInstances}`;
  const controller = new AbortController();
  let mode = 'pretrain';
  let selected = 'input';
  let expandedLayer = 0;
  const escaped = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  const modes = {
    pretrain: { title: '遮挡预训练', path: '遮挡 STFT → 主干 → 重建头 → masked L1', result: '重建时频特征 [B,T,40]', loss: 'L = mean_M |y − ŷ| + 2 · mean_{M ∩ |y|>1} |y − ŷ|', note: '只在 mask_label 选中的时频单元计算损失。两个 L1 分别求均值；高幅值阈值作用于标准化目标。', count: '43,183,144', countLabel: '主干 + 重建头参数', exit: 'reconstruction' },
    embedding: { title: '提取表示', path: 'STFT → 主干 → 直接返回 h', result: '上下文表示 h [B,T,768]', loss: '本次提取不计算 loss', note: '设置 intermediate_rep=True，在重建头之前返回。每帧一个表示；没有自动把整段信号池化成一个向量。通常配合 eval() 与 no_grad()。', count: '42,560,256', countLabel: '本条路径使用的主干参数', exit: 'embedding' },
    classification: { title: '下游分类', path: 'STFT → 主干 → 中心 10 帧均值 → 分类头 → BCE', result: '分类 logit [B,1]', loss: 'L = BCEWithLogitsLoss(logit, label)', note: '只计算分类损失。本次分类不经过重建头；可冻结主干，也可让分类梯度更新主干。预测概率时再 sigmoid。', count: '42,561,025', countLabel: '主干 + 分类头参数（全量微调）', exit: 'classifier' },
  };
  const sources = {
    stft: {
      title: '先把一维波形变成时频特征', shape: '[B,10240] → [B,186,40]（5 秒样本）', file: 'preprocessors/stft.py', path: 'preprocessors/stft.py',
      why: '同一个时间窗内有哪些频率成分，先由 STFT 显式表示。40 是连续频率特征数；每个时间帧随后成为一个 token。STFT 本身没有可训练参数。',
      code: `# STFTPreprocessor.forward / get_stft（节选）\nf, t, Zxx = signal.stft(\n    wav, 2048, nperseg=400, noverlap=350,\n    return_onesided=True)\nZxx = Zxx[:40]\nZxx = np.abs(Zxx)\nZxx = zscore(Zxx, axis=-1)\nZxx = Zxx[:, 10:-10]\nreturn torch.Tensor(np.transpose(Zxx))`,
      note: '这里列出与训练配置一致的数值。单电极样本 10,240 点得到 186 帧；一般长度用 T 表示。预处理位于模型前向之外。',
    },
    input: {
      title: '输入编码：换维度，也保留时间位置', shape: '[B,T,40] → [B,T,768]', file: 'models/transformer_encoder_input.py', path: 'models/transformer_encoder_input.py',
      why: '线性层把 40 个频率特征映射到 768 维工作空间；固定 sin/cos 位置编码提供顺序；LayerNorm 规范每帧的特征尺度；Dropout 在训练时提供正则化。这里没有离散 token ID 或 embedding 查表。',
      code: `# TransformerEncoderInput.forward\ninput_specs = self.in_proj(input_specs)\n# Linear(40, 768), 含 bias\ninput_specs, pos_enc = (\n    self.positional_encoding(input_specs))\ninput_specs = self.layer_norm(input_specs)\ninput_specs = self.dropout(input_specs)\nreturn input_specs, pos_enc`,
      note: '固定正弦位置编码是 buffer，不是可训练参数。输入 Dropout 的 p=0.1；eval() 时 Dropout 关闭。输入投影与 LayerNorm 合计 33,024 个参数。',
    },
    encoder: {
      title: '同样的结构，串联 6 次', shape: '[T,B,768] → [T,B,768]', file: 'models/masked_tf_model.py', path: 'models/masked_tf_model.py',
      why: '每层先沿时间聚合上下文，再对每帧做非线性变换。6 层是 6 组独立的可训练参数。维度与时间长度保持不变，表示在深度方向逐步更新。',
      code: `# build_model（large 配置）\nencoder_layer = nn.TransformerEncoderLayer(\n    d_model=768, nhead=12,\n    dim_feedforward=3072, activation='gelu')\nself.transformer = nn.TransformerEncoder(\n    encoder_layer, num_layers=6)\n\n# forward\nx = x.transpose(0, 1)  # [B,T,D] → [T,B,D]\nh = self.transformer(x,\n    src_key_padding_mask=pad_mask)\nh = h.transpose(0, 1)  # 回到 [B,T,D]`,
      note: '源码没有传 norm_first，使用 PyTorch 默认 post-norm；默认 dropout=0.1。未额外添加顶层最终 LayerNorm。此处只展示结构，不计算表示或注意力。',
    },
    mha: {
      title: 'MHA：从其他时间帧读取信息', shape: '12 heads × 64 维 → 拼接回 768 维', file: 'nn.TransformerEncoderLayer / self_attn', path: 'models/masked_tf_model.py',
      why: 'Q 与 K 决定当前帧读取哪些帧，V 提供被读取的内容。12 个 head 使用各自的投影；它们的结果拼接并输出投影。注意力矩阵两条轴都是时间，不是频率。',
      code: `# PyTorch 默认 post-norm 计算顺序（等价示意）\nattn = self.self_attn(\n    x, x, x,\n    key_padding_mask=pad_mask,\n    need_weights=False)[0]\n# head: Q,K,V 各 [B,T,64]\n# A = softmax(Q @ K.T / sqrt(64) + mask)\n# attention dropout p=0.1（训练时）\nattn = self.dropout1(attn)`,
      note: '输入/输出均为 [T,B,768]。padding mask 的形状为 [B,T]；True 排除对应 key。仓库的该路径没有因果 mask。注意力自身与输出残差分支都使用默认 dropout。',
    },
    norm1: {
      title: '第一次残差与归一化', shape: 'x₁ = LayerNorm(x + Dropout(MHA(x)))', file: 'nn.TransformerEncoderLayer / norm1', path: 'models/masked_tf_model.py',
      why: '残差提供一条保留原表示与传播梯度的直接路径；LayerNorm 沿每个 token 的 768 个特征维度归一化。post-norm 意味着先相加，再归一化。',
      code: `# PyTorch 默认 post-norm 顺序（等价示意）\nattn = self.self_attn(\n    x, x, x,\n    key_padding_mask=pad_mask,\n    need_weights=False)[0]\nx1 = self.norm1(x + self.dropout1(attn))\n# [T,B,768] + [T,B,768] → [T,B,768]`,
      note: '这是一层内部的第一个 LayerNorm。不要与进入 Transformer 之前的输入 LayerNorm 混淆。',
    },
    ffn: {
      title: 'FFN：逐帧做非线性特征变换', shape: '768 → 3072 → 768（T 与 B 不变）', file: 'nn.TransformerEncoderLayer / linear1, linear2', path: 'models/masked_tf_model.py',
      why: '注意力负责不同时间帧之间的信息交换，FFN 负责每帧内部的特征组合。先扩大到 3072 维，经 GELU 后投影回 768 维，才能接回原来的残差路径。同一层的 FFN 在所有时间帧共享参数。',
      code: `# PyTorch FFN 分支（等价示意）\nff = self.linear1(x1)       # 768 → 3072\nff = self.activation(ff)    # GELU\nff = self.dropout(ff)       # p=0.1\nff = self.linear2(ff)       # 3072 → 768\nff = self.dropout2(ff)      # p=0.1\n# 然后与 x1 相加，再做 norm2`,
      note: 'FFN 不是沿时间卷积，也不会改变 token 数。这里的 3072 是中间特征维度，与 STFT 的时间帧数无关。',
    },
    norm2: {
      title: '第二次残差与归一化', shape: 'x₂ = LayerNorm(x₁ + Dropout(FFN(x₁)))', file: 'nn.TransformerEncoderLayer / norm2', path: 'models/masked_tf_model.py',
      why: '将 FFN 提供的变化加回 x₁，再沿每个 token 的特征维度归一化。得到的 x₂ 进入下一层；第 6 层输出就是提取表示时返回的上下文 embedding。',
      code: `# PyTorch 默认 post-norm 顺序（等价示意）\nff = self.linear2(\n    self.dropout(\n        self.activation(self.linear1(x1))))\nx2 = self.norm2(x1 + self.dropout2(ff))\n# 下一层继续接收 [T,B,768]`,
      note: '每层有自己的 norm1/norm2，以及自己的注意力和 FFN 参数；并非反复调用同一组共享权重。',
    },
    embedding: {
      title: '出口一：直接返回上下文 embedding', shape: 'h: [B,T,768]', file: 'models/masked_tf_model.py', path: 'models/masked_tf_model.py',
      why: '保留每个时间帧的上下文表示，供分类、检索等后续任务使用。提取 embedding 是一个前向计算出口，不需要为了“生成 embedding”额外计算一种 loss。',
      code: `# MaskedTFModel.forward\noutput_specs = self.transformer(\n    input_specs,\n    src_key_padding_mask=src_key_mask)\noutput_specs = output_specs.transpose(0, 1)\nif intermediate_rep:\n    return output_specs  # [B,T,768]`,
      note: '调用方式：model(inputs, pad_mask, intermediate_rep=True)。默认 rep_from_layer=-1；仓库没有实现其他 rep_from_layer 分支。',
    },
    reconstruction: {
      title: '出口二：把表示映射回时频空间', shape: '[B,T,768] → [B,T,40]', file: 'models/spec_prediction_head.py', path: 'models/spec_prediction_head.py',
      why: 'embedding 与目标时频特征的维度不同。重建头把 768 维表示解码回每帧 40 个连续数值，才能与原始目标比较。它是预测时频特征，不是直接生成原始电压波形。',
      code: `# SpecPredictionHead.forward\nh = self.hidden_layer(hidden)  # 768 → 768\nh = self.act_fn(h)             # GELU\nh = self.layer_norm(h)\nh = self.output(h)             # 768 → 40\nreturn h`,
      note: '该头包含 622,888 个参数。提取表示与下游分类均跳过此头，因此不能把它也算作分类训练中有梯度的参数。',
    },
    classifier: {
      title: '出口三：中心池化后做二分类', shape: '[B,T,768] → [B,768] → [B,1]', file: 'models/finetune_model.py', path: 'models/finetune_model.py',
      why: '一个窗口有 T 个表示，而当前二分类任务每个窗口只需要一个 logit。源码取中心 10 帧求平均，再接一个可训练的 Linear(768,1)。均值池化本身没有参数。',
      code: `# FinetuneModel.forward\nh = self.upstream(inputs, pad_mask,\n    intermediate_rep=True)\nmiddle = int(h.shape[1] / 2)\npooled = h[:, middle-5:middle+5].mean(axis=1)\nlogits = self.linear_out(pooled)\nreturn logits  # [B,1]`,
      note: '分类头有 769 个参数。此图采用固定长度样本；原代码按 batch 序列中点取中心，不会为每条有效长度重新找中点。',
    },
    loss: {
      title: '训练目标取决于当前任务', shape: '选择一种路径，计算对应的损失', file: 'criterions/', path: 'criterions/pretrain_masked_criterion.py',
      why: '预训练的目标是补全被选中的时频内容；下游分类的目标是对齐标签。它们属于不同阶段、不同出口，不能默认在一次 forward 中同时计算。',
      code: '', note: '',
    },
  };

  const node = (key, title, shape, subtitle = '') => `<button type="button" class="ar-node" data-ar-node="${key}" aria-pressed="false"><span class="ar-node-title">${title}</span><span class="ar-shape">${shape}</span>${subtitle ? `<span class="ar-node-sub">${subtitle}</span>` : ''}</button>`;
  container.innerHTML = `
  <style>
  .ar-lab{--ar-green:#087f69;--ar-ink:#172c40;--ar-muted:#60716f;--ar-line:#e0e7de;--ar-orange:#db7447;color:var(--ar-ink);font-size:15px;line-height:1.7;max-width:1260px;margin:auto}.ar-lab *{box-sizing:border-box}.ar-lab button{font:inherit;cursor:pointer}.ar-lab button:focus-visible,.ar-lab a:focus-visible{outline:3px solid var(--ar-orange);outline-offset:4px}.ar-lab h2{font-size:clamp(23px,2.6vw,31px);line-height:1.3;letter-spacing:-.03em;margin:0 0 8px}.ar-lab h3{font-size:18px;line-height:1.4;margin:0 0 8px;letter-spacing:-.02em}.ar-lab p{margin:0 0 12px}.ar-muted{color:var(--ar-muted)}.ar-small{font-size:13px}.ar-eyebrow{font-size:12px;font-weight:700;color:var(--ar-green);letter-spacing:.12em;margin-bottom:7px}.ar-mono,.ar-shape{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-.03em}.ar-badge{font-size:12px;font-weight:650;background:#eaf4ee;color:var(--ar-green);border-radius:20px;display:inline-block;padding:4px 9px}.ar-badge--orange{background:#fbeee4;color:#a65a32}.ar-intro{margin-bottom:22px;max-width:870px}.ar-card{background:#fff;border:1px solid var(--ar-line);border-radius:16px;padding:23px;min-width:0}.ar-modebar{display:flex;justify-content:space-between;align-items:center;gap:15px;flex-wrap:wrap;margin-bottom:19px}.ar-modes{display:flex;gap:4px;background:#ecefe6;border:1px solid #e2e8dd;border-radius:10px;padding:4px}.ar-mode{border:0;background:transparent;color:var(--ar-muted);border-radius:7px;padding:10px 16px;font-size:14px!important;font-weight:650!important}.ar-mode[aria-pressed=true]{background:#fff;color:var(--ar-green);box-shadow:0 2px 5px #203b2b12}.ar-mode-path{font-size:13px;color:var(--ar-green);padding:11px 15px;background:#edf5ef;border-radius:9px;margin-bottom:20px}.ar-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(300px,1fr);gap:20px;align-items:start}.ar-main{display:flex;flex-direction:column;gap:18px}.ar-map-label{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:14px}.ar-map-label h3{margin:0}.ar-flowline{display:grid;grid-template-columns:1fr 26px 1.5fr;align-items:center;gap:7px}.ar-arrow{color:#89a18d;font-size:21px;text-align:center}.ar-node{background:#f7f8f3;border:1px solid #dfe7dc;border-radius:10px;padding:14px;text-align:left;display:block;width:100%;min-width:0;color:var(--ar-ink);transition:border-color .18s,background .18s,box-shadow .18s}.ar-node:hover{border-color:#83b69d}.ar-node[aria-pressed=true]{box-shadow:0 0 0 2px #087f6929;border-color:var(--ar-green)}.ar-node-title{font-size:15px;line-height:1.4;font-weight:750;display:block}.ar-shape{display:block;font-size:12px;margin-top:6px;color:var(--ar-muted)}.ar-node-sub{font-size:12px;line-height:1.65;display:block;margin-top:6px;color:var(--ar-muted)}.ar-down{text-align:center;color:#87a08e;height:25px;font-size:20px;line-height:25px}.ar-encoder{padding:16px;border:1px solid #c7d9ca;background:#f2f7f0;border-radius:12px}.ar-encoder-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:13px}.ar-encoder-title{font-weight:750;font-size:16px}.ar-plain{background:none;border:0;color:var(--ar-green);padding:0;font-size:12px!important;text-decoration:underline;text-underline-offset:3px}.ar-layer-list{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}.ar-layer{border:1px solid #cbdbca;border-radius:7px;background:white;color:#69806b;padding:10px 4px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace!important;font-size:12px!important}.ar-layer[aria-expanded=true]{color:#fff;background:var(--ar-green);border-color:var(--ar-green)}.ar-layer-detail{background:white;border:1px solid #d8e4d6;border-radius:10px;padding:15px;margin-top:13px}.ar-layer-detail[hidden]{display:none}.ar-layer-caption{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:12px;font-size:12px;color:var(--ar-muted)}.ar-layer-inner{display:grid;grid-template-columns:1fr 1fr;gap:9px}.ar-layer-inner .ar-node{padding:12px}.ar-layer-inner .ar-node-title{font-size:13px}.ar-layer-inner .ar-shape{font-size:11px}.ar-block-number{display:inline-block;color:var(--ar-green);background:#eaf3e7;border-radius:4px;padding:1px 5px;margin-right:5px;font-size:11px}.ar-order-note{font-size:12px;color:var(--ar-muted);padding:11px 0 0;margin:0!important}.ar-exit-label{text-align:center;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--ar-green);font-size:12px;margin:4px 0 13px}.ar-branches{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;position:relative}.ar-branch{position:relative;opacity:.5;transition:opacity .2s}.ar-branch[data-active=true]{opacity:1}.ar-branch::before{content:'↓';display:block;text-align:center;color:#92a78e;font-size:18px;margin-top:-3px;margin-bottom:5px}.ar-branch[data-active=true]::before{color:var(--ar-green)}.ar-branch[data-active=true] .ar-node{background:#edf7ee;border-color:#a3c9b4}.ar-branches .ar-node{padding:12px 10px;height:100%;min-height:155px}.ar-branches .ar-node-title{font-size:13px}.ar-branches .ar-shape{font-size:11px}.ar-branches .ar-node-sub{font-size:11px}.ar-loss{border:1px solid #e8d9c9;background:#fff7ed;border-radius:11px;padding:16px;margin-top:20px}.ar-loss-button{width:100%;display:flex;align-items:center;justify-content:space-between;gap:10px;border:0;background:transparent;padding:0;text-align:left;color:#a55c33;font-size:13px!important;font-weight:700!important}.ar-loss-formula{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:13px;line-height:1.8;color:#a55c33;overflow-wrap:anywhere;margin-top:8px}.ar-loss-note{font-size:12px;color:#95745b;margin-top:8px!important;margin-bottom:0!important}.ar-source{position:sticky;top:20px;overflow:hidden;padding:0}.ar-source-text{padding:22px 22px 17px}.ar-source-text h3{margin-top:8px;font-size:20px}.ar-source-shape{font-size:12px;color:var(--ar-green);margin:12px 0 14px}.ar-source-file{display:flex;align-items:center;justify-content:space-between;gap:15px;background:#193a35;color:#b5cfc2;padding:12px 18px;font-size:11px;overflow-wrap:anywhere}.ar-source-file a{color:#c4dfd3;text-decoration:none;flex-shrink:0}.ar-source-code{background:#203e38;color:#e0ece0;margin:0;padding:19px;font-size:12px;line-height:1.9;white-space:pre-wrap;overflow-wrap:anywhere;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;min-height:195px}.ar-source-note{background:#f4f6ef;color:var(--ar-muted);padding:16px 21px;font-size:12px;line-height:1.8;margin:0!important}.ar-budget{margin-top:20px;display:grid;grid-template-columns:1.15fr 1fr;gap:20px}.ar-count{font-size:clamp(26px,3vw,36px);color:var(--ar-green);font-weight:750;letter-spacing:-.04em;line-height:1.2;margin:10px 0 8px}.ar-count-caption{font-size:12px;color:var(--ar-muted)}.ar-table{width:100%;border-collapse:collapse;font-size:12px;margin-top:13px}.ar-table td{border-top:1px solid var(--ar-line);padding:9px 0}.ar-table td:last-child{text-align:right;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;color:var(--ar-ink)}.ar-table tr[data-active=true] td{color:var(--ar-green);font-weight:700}.ar-facts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:13px}.ar-fact{padding:12px;background:#f4f7f0;border-radius:8px}.ar-fact strong{display:block;color:var(--ar-green);font-size:20px;line-height:1.3}.ar-fact span{font-size:12px;color:var(--ar-muted)}.ar-footer{font-size:12px;line-height:1.85;color:var(--ar-muted);border-top:1px solid var(--ar-line);margin-top:19px;padding-top:14px}.ar-footer a{color:var(--ar-green)}
  @media(max-width:1100px){.ar-grid{grid-template-columns:minmax(0,1.4fr) minmax(280px,1fr);gap:16px}.ar-card{padding:18px}.ar-source{padding:0}.ar-branches{gap:6px}.ar-branches .ar-node{padding:11px 8px}.ar-branches .ar-node-title{font-size:12px}.ar-layer-inner .ar-node{padding:11px}.ar-source-code{padding:16px;font-size:11px}.ar-layer-detail{padding:11px}.ar-budget{gap:16px}}
  @media(max-width:900px){.ar-grid{grid-template-columns:1fr}.ar-source{position:static}.ar-source-code{min-height:0;font-size:12px}.ar-budget{grid-template-columns:1fr 1fr}}
  @media(max-width:580px){.ar-modebar{align-items:stretch}.ar-modes{display:flex;width:100%}.ar-mode{padding:10px 8px;flex:1;font-size:12px!important}.ar-card{padding:16px;border-radius:12px}.ar-source{padding:0}.ar-flowline{grid-template-columns:1fr;gap:4px}.ar-arrow{transform:rotate(90deg);line-height:24px}.ar-layer-list{gap:4px}.ar-layer{padding:10px 1px;font-size:11px!important}.ar-encoder{padding:12px}.ar-branches{grid-template-columns:1fr}.ar-branches .ar-node{min-height:0;padding:13px}.ar-branches .ar-node-title{font-size:14px}.ar-branches .ar-shape{font-size:12px}.ar-branches .ar-node-sub{font-size:12px}.ar-branch::before{margin:1px 0 4px}.ar-budget{grid-template-columns:1fr;gap:16px}.ar-layer-inner{grid-template-columns:1fr}.ar-layer-inner .ar-node-title{font-size:14px}.ar-layer-inner .ar-shape{font-size:12px}.ar-encoder-head{align-items:flex-start}.ar-source-text{padding:19px}.ar-count{font-size:31px}}
  @media(prefers-reduced-motion:reduce){.ar-lab *{transition:none!important}}
  </style>
  <section class="ar-lab" aria-labelledby="${uid}-title">
    <div class="ar-intro"><p class="ar-eyebrow">MODEL ARCHITECTURE / LARGE</p><h2 id="${uid}-title">一套主干，三个出口。</h2><p class="ar-muted">点击模块读源码，展开任意一层看内部计算。切换任务，跟踪同一段脑电在不同阶段的路径。</p></div>
    <div class="ar-modebar"><div class="ar-modes" role="group" aria-label="选择模型工作路径">${Object.entries(modes).map(([key, value])=>`<button type="button" class="ar-mode" data-ar-mode="${key}" aria-pressed="${key===mode}">${value.title}</button>`).join('')}</div><span class="ar-badge ar-badge--orange">结构讲解 · 不执行模型推理</span></div>
    <div class="ar-mode-path" data-ar="path" aria-live="polite"></div>
    <div class="ar-grid">
      <div class="ar-main"><div class="ar-card">
        <div class="ar-map-label"><h3>从时频特征开始</h3><span class="ar-badge">T = 186 · 示例</span></div>
        <div class="ar-flowline">${node('stft','STFT 特征','[B,T,40]','输入来自单电极连续波形')}<span class="ar-arrow" aria-hidden="true">→</span>${node('input','输入编码','[B,T,40] → [B,T,768]','Linear + sin/cos PE + LayerNorm + Dropout')}</div>
        <div class="ar-down" aria-hidden="true">↓</div>
        <div class="ar-encoder"><div class="ar-encoder-head"><div><span class="ar-encoder-title">TransformerEncoder × 6</span><div class="ar-shape">转为 [T,B,768] → 逐层处理 → 转回</div></div><button type="button" class="ar-plain" data-ar-node="encoder" aria-pressed="false">查看构造代码 ↗</button></div>
          <div class="ar-layer-list" role="group" aria-label="展开 Transformer 层">${Array.from({length:6},(_,i)=>`<button type="button" class="ar-layer" data-ar-layer="${i}" aria-expanded="${i===0}" aria-controls="${uid}-layer-detail">层 ${i+1}</button>`).join('')}</div>
          <div class="ar-layer-detail" id="${uid}-layer-detail" data-ar="layer-detail"><div class="ar-layer-caption"><span data-ar="layer-caption">第 1 层内部</span><span>Post-norm · 结构逐层相同</span></div><div class="ar-layer-inner">
            ${node('mha','① 多头自注意力','12 × 64 → 768','Q/K/V → softmax → 加权汇总 → 输出投影')}
            ${node('norm1','② 残差 + LayerNorm','LN(x + Dropout(MHA(x)))','先相加，再做归一化')}
            ${node('ffn','③ FFN + GELU','768 → 3072 → 768','Linear → GELU → Dropout → Linear')}
            ${node('norm2','④ 残差 + LayerNorm','LN(x₁ + Dropout(FFN(x₁)))','输出进入下一层，形状保持不变')}
          </div><p class="ar-order-note">按 ① → ② → ③ → ④ 计算。每层使用独立参数；展开层只切换说明，不是只运行该层。</p></div>
        </div>
        <div class="ar-down" aria-hidden="true">↓</div><div class="ar-exit-label">上下文表示 h · [B,T,768]</div>
        <div class="ar-branches">
          <div class="ar-branch" data-ar-branch="embedding">${node('embedding','直接取 embedding','返回 [B,T,768]','intermediate_rep=True；此处不需要 loss')}</div>
          <div class="ar-branch" data-ar-branch="reconstruction">${node('reconstruction','重建预测头','768 → 768 → 40','Linear → GELU → LN → Linear；逐帧预测时频特征')}</div>
          <div class="ar-branch" data-ar-branch="classifier">${node('classifier','中心池化 + 分类头','[B,T,768] → [B,1]','中心 10 帧 mean → Linear(768,1)；新增二分类头')}</div>
        </div>
        <div class="ar-loss"><button type="button" class="ar-loss-button" data-ar-node="loss" aria-pressed="false"><span data-ar="loss-title"></span><span aria-hidden="true">查看源码 ↗</span></button><div class="ar-loss-formula" data-ar="loss-formula"></div><p class="ar-loss-note" data-ar="loss-note"></p></div>
      </div></div>
      <aside class="ar-card ar-source" aria-labelledby="${uid}-source-title"><div class="ar-source-text"><div class="ar-eyebrow">CLICK A MODULE / WHY IT EXISTS</div><h3 id="${uid}-source-title" data-ar="source-title"></h3><div class="ar-source-shape ar-mono" data-ar="source-shape"></div><p class="ar-small ar-muted" data-ar="source-why"></p></div><div class="ar-source-file"><span class="ar-mono" data-ar="source-file"></span><a data-ar="source-link" target="_blank" rel="noopener noreferrer">仓库 ↗</a></div><pre class="ar-source-code" data-ar="source-code"></pre><p class="ar-source-note" data-ar="source-note"></p></aside>
    </div>
    <div class="ar-budget"><div class="ar-card"><p class="ar-eyebrow">PARAMETER BUDGET</p><h3>数清楚，哪一部分属于主干。</h3><div class="ar-count ar-mono" data-ar="count"></div><div class="ar-count-caption" data-ar="count-label"></div><table class="ar-table" aria-label="large 模型参数组成"><tbody>
      <tr><td>输入投影 + 输入 LayerNorm</td><td>33,024</td></tr><tr><td>6 层 TransformerEncoder</td><td>42,527,232</td></tr><tr data-ar-budget="backbone"><td>主干合计</td><td>42,560,256</td></tr><tr data-ar-budget="reconstruction"><td>重建预测头</td><td>622,888</td></tr><tr data-ar-budget="classifier"><td>新增分类头</td><td>769</td></tr>
    </tbody></table><p class="ar-small ar-muted" style="margin:12px 0 0">按 large 架构与含 bias 的默认层计算，不含固定位置编码 buffer。这里统计路径参数；冻结后的可训练参数见微调章节。</p></div>
    <div class="ar-card"><p class="ar-eyebrow">THE IMPORTANT DEFAULTS</p><h3>读源码时，这几个默认值别漏掉。</h3><div class="ar-facts"><div class="ar-fact"><strong>Post-norm</strong><span>未传 norm_first；相加后归一化</span></div><div class="ar-fact"><strong>p = 0.1</strong><span>输入与 encoder 的 Dropout；eval 时关闭</span></div><div class="ar-fact"><strong>12 × 64</strong><span>12 个注意力 head，合计 768 维</span></div><div class="ar-fact"><strong>无 causal mask</strong><span>可读取前后帧；padding 只排除 key</span></div></div><p class="ar-small ar-muted" style="margin:16px 0 0">Transformer 内部使用 [T,B,D]，模型接口使用 [B,T,D]。转置只改变轴顺序，不会把时间帧变成特征维度。</p><p class="ar-small ar-muted">重建 L1 和分类 BCE 对应各自路径；提取表示时直接返回 h。</p></div></div>
    <div class="ar-footer">依据本地仓库 <span class="ar-mono">masked_tf_model.py</span>、<span class="ar-mono">transformer_encoder_input.py</span>、<span class="ar-mono">spec_prediction_head.py</span> 与 <span class="ar-mono">finetune_model.py</span>。Encoder 内部代码为 PyTorch 默认 post-norm 的等价计算顺序；没有加载预训练权重，也不生成预测。</div>
  </section>`;

  const get = key => container.querySelector(`[data-ar="${key}"]`);
  const on = (element, name, handler) => element.addEventListener(name, handler, { signal: controller.signal });
  const allNodes = () => [...container.querySelectorAll('[data-ar-node]')];

  function showSource(key) {
    selected = key;
    const base = sources[key];
    let source = base;
    if (key === 'loss') {
      source = mode === 'pretrain' ? {
        ...base, title: '遮挡重建：L1 + 高幅值加权 L1', file: 'criterions/pretrain_masked_criterion.py', path: 'criterions/pretrain_masked_criterion.py', shape: '只监督 mask_label=True 的时频单元',
        code: `# PretrainMaskedCriterion.forward（节选）\nmask = batch['mask_label'].bool()\ntrue_activity = labels.masked_select(mask)\npredicted = output.masked_select(mask)\nl1 = (true_activity - predicted).abs().mean()\nhigh = true_activity.abs() > 1\ncontent_l1 = (\n    true_activity[high] - predicted[high]\n).abs().mean()\nloss = l1 + self.cfg.alpha * content_l1\n# alpha = 2（默认配置）`,
        note: '此 loss 同时训练输入编码、Transformer 与重建头。这里不计算分类 BCE。源码在高幅值子集为空时会出现空均值，需要实际训练时留意。',
      } : mode === 'classification' ? {
        ...base, title: '二分类：直接对 logit 算 BCE', file: 'criterions/finetune_criterion.py', path: 'criterions/finetune_criterion.py', shape: 'logits.squeeze(-1): [B] · labels: [B]',
        code: `# FinetuneCriterion\nself.loss_fn = nn.BCEWithLogitsLoss(\n    reduction='mean')\n\noutput = model(inputs, pad_mask)\nlabels = torch.FloatTensor(batch['labels'])\noutput = output.squeeze(-1)\nloss = self.loss_fn(output, labels)\n# 概率仅用于预测：sigmoid(output)`,
        note: 'BCEWithLogitsLoss 内部处理 sigmoid；训练前不要再次手动 sigmoid。是否更新主干由 frozen_upstream 决定。',
      } : {
        ...base, title: '提取 embedding 不需要 loss', file: 'models/masked_tf_model.py', path: 'models/masked_tf_model.py', shape: '直接返回 h: [B,T,768]',
        code: `model.eval()\nwith torch.no_grad():\n    h = model(inputs, pad_mask,\n        intermediate_rep=True)\n# 不调用重建头；不调用 criterion\n# h: [B,T,768]`,
        note: '模型参数来自先前的预训练。现在只计算表示，不表示 embedding 从未受 loss 训练。',
      };
    }
    get('source-title').textContent = source.title;
    get('source-shape').textContent = source.shape;
    get('source-why').textContent = source.why;
    get('source-file').textContent = source.file;
    get('source-code').textContent = source.code;
    get('source-note').textContent = source.note;
    get('source-link').setAttribute('href', `https://github.com/czlwang/brainbert/blob/master/${source.path}`);
    allNodes().forEach(el=>el.setAttribute('aria-pressed', String(el.dataset.arNode === key)));
  }

  function updateMode() {
    const current = modes[mode];
    container.querySelectorAll('[data-ar-mode]').forEach(el=>el.setAttribute('aria-pressed', String(el.dataset.arMode === mode)));
    container.querySelectorAll('[data-ar-branch]').forEach(el=>el.dataset.active=String(el.dataset.arBranch===current.exit));
    get('path').textContent = `${current.title} · ${current.path}`;
    get('loss-title').textContent = mode === 'embedding' ? '提取阶段 · 无训练损失' : `${current.title} · 当前目标`;
    get('loss-formula').textContent = current.loss;
    get('loss-note').textContent = current.note;
    get('count').textContent = current.count;
    get('count-label').textContent = current.countLabel;
    container.querySelectorAll('[data-ar-budget]').forEach(el=>el.dataset.active=String(el.dataset.arBudget==='backbone'||el.dataset.arBudget===current.exit));
    if(selected==='loss')showSource('loss');
  }

  allNodes().forEach(el=>on(el, 'click', ()=>showSource(el.dataset.arNode)));
  container.querySelectorAll('[data-ar-mode]').forEach(el=>on(el, 'click', ()=>{mode=el.dataset.arMode;updateMode();showSource(modes[mode].exit);}));
  container.querySelectorAll('[data-ar-layer]').forEach(el=>on(el, 'click', ()=>{
    const index=Number(el.dataset.arLayer);expandedLayer=expandedLayer===index?-1:index;
    container.querySelectorAll('[data-ar-layer]').forEach(button=>button.setAttribute('aria-expanded',String(Number(button.dataset.arLayer)===expandedLayer)));
    get('layer-detail').hidden=expandedLayer<0;
    if(expandedLayer>=0){get('layer-caption').textContent=`第 ${expandedLayer+1} 层内部`;showSource('encoder');}
  }));
  updateMode();
  showSource('input');
  return ()=>{controller.abort();container.replaceChildren();};
}
