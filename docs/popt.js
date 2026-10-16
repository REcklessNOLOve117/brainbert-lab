import { populationAttention, defaultCoordinates, binaryCrossEntropy, replacementLoss } from './popt-math.js';

// Pin source links to the commit inspected for this lesson.
const REPO = 'https://github.com/czlwang/PopulationTransformer/blob/d237755bbdea123c60bface0cecbb48bab6f42de/';
let instances = 0;

const labelName = index => index === 0 ? 'CLS' : `E${String(index).padStart(2, '0')}`;
const number = value => Number(value).toFixed(4);
const sigmoid = value => 1 / (1 + Math.exp(-value));
const labelText = value => ['忽略', '替换', '保留'][value];

export function mountPopT(container) {
  const uid = `pt-${++instances}`;
  const controller = new AbortController();
  let count = 8;
  let coordinates = defaultCoordinates(count);
  let query = 1;
  let key = 2;
  let head = 0;
  let splitSequence = true;
  let usePosition = true;
  let selectedModule = 'brainbert';
  let expandedLayer = 0;
  let route = 'pretrain';
  let pairLabel = 0;
  let pairLogit = 0.7;
  let tokenLabels = Array.from({ length: count }, (_, i) => [1, 2, 0, 2][i % 4]);
  let tokenLogits = Array.from({ length: count }, (_, i) => Math.round(Math.sin(i * 1.37 + 0.4) * 20) / 10);
  let selectedToken = 0;
  let fineLabel = 1;
  let fineLogit = 0.9;
  let frozen = false;
  let attention;

  const node = (id, title, shape, note = '') => `<button type="button" class="pt-node" data-pt-module="${id}" aria-pressed="false"><strong>${title}</strong><span class="pt-mono">${shape}</span>${note ? `<span>${note}</span>` : ''}</button>`;

  container.innerHTML = `
  <section class="pt-lab" aria-labelledby="${uid}-title">
    <div class="pt-intro"><div class="pt-kicker">POPULATION TRANSFORMER / ICLR 2025</div><h2 id="${uid}-title">从一根电极，走向一组神经活动。</h2><p class="pt-muted">BrainBERT 沿时间理解单电极；PopT 再把同一窗口的多电极表示组织起来，在电极之间交换信息。这里的 token 是<strong>电极表示</strong>，不再是一个 STFT 时间帧。</p><div class="pt-badges"><span class="pt-badge">BrainBERT 768 → PopT 512</span><span class="pt-badge">6 层 · 8 heads · 跨电极注意力</span><span class="pt-badge pt-badge-warm">教学计算 · 未加载 .pth</span></div></div>

    <section class="pt-section" aria-labelledby="${uid}-architecture"><div class="pt-section-head"><div><div class="pt-step">01 / CLICK THE ARCHITECTURE</div><h3 id="${uid}-architecture">两级编码，各自解决一个问题。</h3><p class="pt-muted pt-small">点击模块查看形状和对应源码；展开任意一层，检查残差与归一化的先后顺序。</p></div><span class="pt-badge">N = 电极数 · B = batch</span></div>
      <div class="pt-grid"><div class="pt-card">
        <div class="pt-flow">${node('brainbert', '① 每电极 BrainBERT', '[B,N,T,768] → [B,N,768]', '中心 10 帧 mean · 离线冻结')}<span class="pt-arrow" aria-hidden="true">→</span>${node('cls', '② 在最前面加 CLS', '[B,N+1,768]', 'CLS 原始输入固定为全 1 向量')}</div>
        <div class="pt-center-arrow" aria-hidden="true">↓</div>
        <div class="pt-flow">${node('projection', '③ 输入线性投影', '768 → 512', '所有 token 共用一个 Linear')}<span class="pt-arrow" aria-hidden="true">＋</span>${node('position', '④ 空间 + 组别编码', 'L128 ⊕ I128 ⊕ P128 ⊕ S128', '固定 sin/cos · 拼成 512 维再相加')}</div>
        <div class="pt-center-arrow" aria-hidden="true">↓</div>
        <div class="pt-flow pt-single">${node('normalization', '⑤ 输入 LayerNorm + Dropout', '[B,N+1,512]', '沿特征维归一化；训练时 dropout p=0.1')}</div>
        <div class="pt-center-arrow" aria-hidden="true">↓</div>
        <div class="pt-encoder"><div class="pt-encoder-head">${node('encoder', '⑥ TransformerEncoder × 6', '8 heads × 64 · FFN 512 → 2048 → 512', '每层独立参数 · GELU · post-LN')}</div><div class="pt-layers" role="group" aria-label="展开 PopT Transformer 层">${Array.from({ length: 6 }, (_, i) => `<button type="button" data-pt-layer="${i}" aria-expanded="${i === 0}" aria-controls="${uid}-layer">层 ${i + 1}</button>`).join('')}</div><div id="${uid}-layer" class="pt-layer-detail" data-pt="layer-detail"><div class="pt-small pt-muted" data-pt="layer-caption">第 1 层 · 先残差相加，再 LayerNorm</div><div class="pt-layer-step"><span>① 电极之间交换信息</span><code>MHA(x) → Dropout</code></div><div class="pt-layer-step"><span>② 第一次残差与归一化</span><code>x₁ = LN(x + attn)</code></div><div class="pt-layer-step"><span>③ 每个 token 内部变换</span><code>Linear → GELU → Dropout → Linear</code></div><div class="pt-layer-step"><span>④ 第二次残差与归一化</span><code>x₂ = LN(x₁ + Dropout(ff))</code></div></div></div>
        <div class="pt-center-arrow" aria-hidden="true">↓</div><div class="pt-exits">${node('pretrain', '预训练：两项判别任务', 'CLS → 1；各电极 token → 1', '连续 / 随机配对 + 是否被替换')}${node('downstream', '下游：读取全组表示', 'h[:,0,:] → Linear(512,1)', '取 CLS，预测有标签的分类目标')}</div>
        <p class="pt-note">PopT 的注意力轴是 N+1 个 token。电极数量可以改变；输入的三维坐标用于说明“这个 token 来自哪里”，seq_id 用于说明“属于哪一组”。</p>
      </div><aside class="pt-card pt-source" aria-labelledby="${uid}-source-title"><div class="pt-kicker">SOURCE / SHAPES / PURPOSE</div><h4 id="${uid}-source-title" data-pt="source-title"></h4><div class="pt-source-shape pt-mono" data-pt="source-shape"></div><p class="pt-muted pt-small" data-pt="source-explanation"></p><pre data-pt="source-code"></pre><p class="pt-muted pt-small" data-pt="source-note"></p><div class="pt-source-link"><span class="pt-mono" data-pt="source-file"></span><a data-pt="source-link" target="_blank" rel="noopener noreferrer">核对源码 ↗</a></div></aside></div>
    </section>

    <section class="pt-section" aria-labelledby="${uid}-experiment"><div class="pt-section-head"><div><div class="pt-step">02 / MULTI-ELECTRODE EXPERIMENT</div><h3 id="${uid}-experiment">当前电极，向谁读取信息？</h3><p class="pt-muted pt-small">点击电极选 query；点击矩阵单元选 query / key。改变坐标、位置编码或 head，观察 softmax 与加权汇总重新计算。</p></div></div>
      <div class="pt-controls"><label class="pt-control" for="${uid}-count">电极数<select id="${uid}-count" data-pt="count"><option value="4">4 根</option><option value="8" selected>8 根</option><option value="12">12 根</option></select></label><label class="pt-control" for="${uid}-head">注意力 head<select id="${uid}-head" data-pt="head">${Array.from({ length: 8 }, (_, i) => `<option value="${i}">Head ${i + 1}</option>`).join('')}</select></label><label class="pt-check"><input type="checkbox" data-pt="split" checked> 两组输入：seq_id = 0 / 1</label><label class="pt-check"><input type="checkbox" data-pt="position" checked> 加入空间 / 组别 PE</label></div>
      <div class="pt-experiment"><div class="pt-map"><h4>电极位置 · 二维布局示意</h4><p class="pt-small pt-muted">圆点可点击；线的深浅表示当前 query 的权重。</p><div class="pt-map-graphic" data-pt="map"></div><button type="button" class="pt-query-cls" data-pt-query="0" aria-pressed="false">选 CLS 作为 query</button><div class="pt-coord-heading" data-pt="coordinate-title"></div><div class="pt-coords">${['L', 'I', 'P'].map((axis, i) => `<label class="pt-coordinate" for="${uid}-coord-${axis}"><span>${axis}</span><input type="range" id="${uid}-coord-${axis}" data-pt-coordinate="${i}" min="0" max="400" step="1" value="100"><output data-pt="coordinate-${i}"></output></label>`).join('')}</div><p class="pt-small pt-muted" style="margin:12px 0 0">使用非负整数示例坐标查 sin/cos 表。图中只投影 L / I；调整 P 仍会改变注意力。这不是脑部解剖图。</p></div>
        <div class="pt-card"><div class="pt-axis-caption"><span>行 = query · 列 = key</span><span>小<span class="pt-scale"></span>大</span></div><div class="pt-matrix-scroll"><div class="pt-matrix" data-pt="matrix" aria-label="电极注意力权重矩阵"></div></div><div class="pt-reading" aria-live="polite"><div><span data-pt="read-pair"></span><strong data-pt="read-weight"></strong></div><div><span>所选 query 的整行权重和</span><strong data-pt="row-sum"></strong></div><div><span data-pt="read-score-label"></span><strong data-pt="read-score"></strong></div></div><div class="pt-formula pt-mono">Q / K / V = xW，d_head = 64<br>Aᵢⱼ = softmaxⱼ(Qᵢ·Kⱼ / √64)<br><strong>cᵢ = Σⱼ Aᵢⱼ Vⱼ</strong> · <span data-pt="context-first"></span></div><div class="pt-vector-bars" data-pt="context-bars" aria-label="加权汇总向量的前16维"></div><p class="pt-small pt-muted">上图是当前 head 的 64 维汇总，条形只画前 16 维。正式 MHA 将 8 个 head 拼回 512 维，再做输出投影。</p></div></div>
      <div class="pt-card" style="margin-top:16px"><h4 data-pt="pe-title"></h4><div class="pt-pe">${['L', 'I', 'P', 'seq_id'].map((axis, i) => `<div class="pt-pe-segment"><strong>${axis} · 128 维</strong><span data-pt="pe-value-${i}"></span></div>`).join('')}</div><div class="pt-formula pt-mono">PE(p, 2j) = sin(p · 10000^(−2j/128))<br>PE(p, 2j+1) = cos(p · 10000^(−2j/128))<br>p_electrode = PE(L) ⊕ PE(I) ⊕ PE(P) ⊕ PE(seq_id)</div><p class="pt-small pt-muted" data-pt="pe-first"></p></div>
      <p class="pt-note"><strong>计算边界：</strong>合成 768 维输入 + 固定种子的随机投影；输入投影、512 维空间编码、LayerNorm、所选 head 的 Q/K/V、softmax 和汇总均实际计算。这里只算一层中的一个 head，不执行完整 6 层、FFN、输出投影或训练；dropout 关闭。权重不是预训练模型注意力，连线不代表真实脑连接性。</p>
    </section>

    <section class="pt-section" aria-labelledby="${uid}-tasks"><div class="pt-section-head"><div><div class="pt-step">03 / PRETRAIN → FINE-TUNE</div><h3 id="${uid}-tasks">预训练学关系，下游才用任务标签。</h3><p class="pt-muted pt-small">切换出口，检查标签、哪些 token 计入 loss，以及分类梯度更新哪一段。</p></div><div class="pt-toggle" role="group" aria-label="PopT 训练路径"><button type="button" data-pt-route="pretrain" aria-pressed="true">两项预训练任务</button><button type="button" data-pt-route="downstream" aria-pressed="false">下游 CLS 分类</button></div></div>
      <div data-pt="pretrain-panel"><div class="pt-task-grid"><div class="pt-card"><div class="pt-kicker">TASK A / ENSEMBLE DECODING</div><h4>这两组来自连续片段吗？</h4><div class="pt-pair"><div class="pt-segment"><b>A</b>第一组电极 · seq_id 0</div><span aria-hidden="true">→</span><div class="pt-segment" data-pt="pair-b"><b>B</b><span data-pt="pair-time"></span></div></div><div class="pt-toggle" role="group" aria-label="片段配对标签"><button type="button" data-pt-pair="0" aria-pressed="true">连续片段 · y=0</button><button type="button" data-pt-pair="1" aria-pressed="false">随机配对 · y=1</button></div><p class="pt-small pt-muted" style="margin-top:12px">官方标签方向：1 表示随机配对。A/B 使用互不重叠的电极子集；CLS 汇总全部 token 后接 Linear(512,1)。</p><label class="pt-small" for="${uid}-pair-logit">调节假设的 CLS logit z（教学数值）</label><div class="pt-logit-control"><input type="range" id="${uid}-pair-logit" data-pt="pair-logit" min="-4" max="4" step="0.1" value="0.7"><output data-pt="pair-logit-value"></output></div><div class="pt-formula pt-mono" data-pt="pair-result"></div></div>
        <div class="pt-card"><div class="pt-kicker">TASK B / CHANNEL DECODING</div><h4>哪些电极 token 被别的片段替换了？</h4><p class="pt-small pt-muted">点选 token，再指定内部标签和假设 logit。忽略项保留在注意力输入里，但不进入这项 BCE。</p><div class="pt-token-picker" data-pt="token-picker" role="group" aria-label="选择用于检查替换损失的电极"></div><div class="pt-toggle" role="group" aria-label="所选 token 的替换标签"><button type="button" data-pt-token-label="0">0 · 不计 loss</button><button type="button" data-pt-token-label="1">1 · 已替换</button><button type="button" data-pt-token-label="2">2 · 保留负例</button></div><p class="pt-small pt-muted" data-pt="token-label-description" style="margin-top:10px"></p><label class="pt-small" for="${uid}-token-logit">调节所选 token 的假设 logit z</label><div class="pt-logit-control"><input type="range" id="${uid}-token-logit" data-pt="token-logit" min="-4" max="4" step="0.1" value="0"><output data-pt="token-logit-value"></output></div><table class="pt-table" aria-label="替换任务的 loss mask 与标签"><thead><tr><th>Token</th><th>内部标签</th><th>BCE y</th><th>logit</th><th>单项 BCE</th></tr></thead><tbody data-pt="token-table"></tbody></table><div class="pt-formula pt-mono" data-pt="replace-result"></div></div></div>
        <div class="pt-loss-total" aria-live="polite"><div><p>当前示例的训练目标</p><span class="pt-mono">L = L_ensemble + L_replace</span></div><strong class="pt-mono" data-pt="total-loss"></strong></div><p class="pt-note">两项均使用 BCEWithLogitsLoss：logit 直接进入损失，展示概率时才 sigmoid。替换 loss 的筛选条件是内部标签为 1 或 2，CLS / 未选 token / padding 不计入该项。这里手动选标签用于观察 loss，不模拟训练采样比例；假设 logits 由滑块提供，与上方随机注意力实验分开。源码虽计算 spec_prediction_head，本 criterion 没有对重建输出施加损失。</p>
      </div>
      <div class="pt-hidden" data-pt="downstream-panel"><div class="pt-card"><div class="pt-kicker">DOWNSTREAM / NEW TASK HEAD</div><h4>读取 CLS，不把所有电极硬拼成固定长度。</h4><div class="pt-fine-flow"><div class="pt-fine-box"><strong>① BrainBERT</strong>每电极中心 10 帧 mean<br><span class="pt-mono">[B,N,768]</span><br>离线 eval + no_grad</div><div class="pt-fine-box" data-pt="fine-backbone"><strong>② 预训练 PopT</strong>多电极 + 坐标 → CLS<br><span class="pt-mono">h[:,0,:] : [B,512]</span><br><span data-pt="fine-backbone-state"></span></div><div class="pt-fine-box" data-train="true"><strong>③ 新增分类头</strong>Linear(512,1)<br><span class="pt-mono">logit : [B,1]</span><br>有监督 BCE · 更新参数</div></div><div class="pt-toggle" role="group" aria-label="是否冻结 PopT 主干"><button type="button" data-pt-freeze="false" aria-pressed="true">更新 PopT + 分类头</button><button type="button" data-pt-freeze="true" aria-pressed="false">冻结 PopT，只训分类头</button></div><p class="pt-gradient" data-pt="gradient-path"></p><div class="pt-task-grid"><div><label class="pt-small" for="${uid}-fine-logit">假设下游分类 logit z</label><div class="pt-logit-control"><input type="range" id="${uid}-fine-logit" data-pt="fine-logit" min="-4" max="4" step="0.1" value="0.9"><output data-pt="fine-logit-value"></output></div><div class="pt-toggle" role="group" aria-label="下游分类真实标签"><button type="button" data-pt-fine-label="0">任务标签 y=0</button><button type="button" data-pt-fine-label="1" aria-pressed="true">任务标签 y=1</button></div></div><div class="pt-formula pt-mono" data-pt="fine-result"></div></div><p class="pt-note">这里的 y 由下游数据集提供，与预训练的“随机配对 / 替换”标签不是同一个含义。官方示例 frozen_upstream=False 更新 PopT 与新分类头；设 True 冻结的是 PopT，前面的 BrainBERT 特征仍已离线提取。冻结会改变梯度流，不改变给定 logit 下的 BCE 数值。</p></div></div>
    </section>
    <footer class="pt-links"><a href="https://arxiv.org/abs/2406.03044" target="_blank" rel="noopener noreferrer">论文 · Population Transformer ↗</a><a href="https://github.com/czlwang/PopulationTransformer" target="_blank" rel="noopener noreferrer">官方源码 ↗</a><a href="https://huggingface.co/PopulationTransformer/popt_brainbert_stft" target="_blank" rel="noopener noreferrer">PopT 预训练权重 ↗</a><a href="https://glchau.github.io/population-transformer/" target="_blank" rel="noopener noreferrer">论文项目页 ↗</a></footer><div class="pt-proof">模型结构、位置编码、标签方向依据官方代码 d237755。配置为 pt_custom_model / multi_subj_position_encoding；交互数值是明确标注的教学实验，不报告真实模型预测或训练效果。</div>
  </section>`;

  const get = name => container.querySelector(`[data-pt="${name}"]`);
  const each = (selector, callback) => container.querySelectorAll(selector).forEach(callback);
  const set = (name, value) => { get(name).textContent = value; };
  const on = (type, handler) => container.addEventListener(type, handler, { signal: controller.signal });

  function showModule(id) {
    selectedModule = id;
    const items = {
      brainbert: { title: 'BrainBERT：先做每根电极的时间表征', shape: `[B,${count},T,768] → [B,${count},768]`, file: 'preprocessors/multi_elec_spec_pretrained.py', line: '#L55-L69', explanation: '每根电极单独经过 BrainBERT，然后取中心 10 个时间帧的均值。PopT 接收的是这个窗口内每根电极的 768 维向量；BrainBERT 的时间轴已经在这一处池化。', code: 'with torch.no_grad():\n    outputs = brainbert(\n        specs, mask, intermediate_rep=True)\n    middle = int(outputs.shape[1] / 2)\n    embeddings = outputs[\n        :, middle-5:middle+5\n    ].mean(axis=1)', note: '官方数据准备阶段先 eval()，再 no_grad() 生成特征。PopT 预训练和下游训练不会通过已保存的特征回传到 BrainBERT。' },
      cls: { title: 'CLS：固定输入，经过网络后成为全组表示', shape: `[B,${count},768] → [B,${count + 1},768]`, file: 'datasets/nsp_replace_only_pretrain.py', line: '#L151-L155', explanation: '源码把一个全 1 的 768 维向量放在第 0 个位置。它不是 nn.Parameter；经过共享投影和 Transformer 后，这个位置能够读取各电极并形成整体表示。', code: 'cls_token = torch.ones(1, embed_dim)\nmasked_inputs = torch.cat([\n    cls_token,\n    masked_inputs_1, masked_inputs_2\n])\n# 第 0 个输出就是 h_CLS', note: 'CLS 的空间 / 组别编码使用 PE(0) 重复 4 份。可训练的投影、注意力和分类头仍会使它的输出随输入与训练改变。' },
      projection: { title: '共享 Linear：768 维进入 512 维工作空间', shape: `[B,${count + 1},768] → [B,${count + 1},512]`, file: 'models/transformer_encoder_input.py', line: '#L66-L84', explanation: '每个电极向量与 CLS 都使用同一层 Linear(768,512)。映射不改变电极数，也不把不同电极拼到特征轴上；可变电极数仍对应可变 token 数。', code: 'self.in_proj = nn.Linear(\n    in_features=cfg.input_dim,  # 768\n    out_features=cfg.hidden_dim # 512\n)\ninput_specs = self.in_proj(input_specs)', note: '该投影是 PopT 可训练参数。其后加入 512 维位置编码，再做输入 LayerNorm 与 Dropout。' },
      position: { title: '固定 sin/cos 表：坐标三段 + 组别一段', shape: '128 + 128 + 128 + 128 = 512', file: 'models/transformer_encoder_input.py', line: '#L28-L62', explanation: 'L / I / P 是电极三维定位坐标。每一轴各查 128 维 sin/cos 表，再拼上 128 维 seq_id。预训练 A/B 两组分别使用 seq_id 0/1；下游单组为 0。它不是电极编号。', code: 'pe_dim = hidden_dim // 4  # 128\np_embed = pe[0, coords].reshape(B,N,384)\nseq_embed = pe[0, seq_id] # [B,N,128]\np = torch.cat([p_embed, seq_embed], -1)\ncls_p = pe[0,0].repeat(B,4)[:,None,:]\nx = x + torch.cat([cls_p,p], axis=1)', note: '源码将坐标转 Long 后直接查 5000 项固定表；没有在此处做可学习 MLP。演示使用 0..400 的非负整数，公式与段顺序按源码。' },
      normalization: { title: '输入归一化，然后进入 Transformer', shape: `[B,${count + 1},512] → [B,${count + 1},512]`, file: 'models/transformer_encoder_input.py', line: '#L76-L84', explanation: '投影向量与位置向量相加后，对每个 token 的 512 维做 LayerNorm，再进行 dropout。此处的 LayerNorm 与每个 Transformer 层内部的两个 LayerNorm 是不同的层。', code: 'input_specs = self.in_proj(input_specs)\ninput_specs, pos_enc = (\n    self.positional_encoding(\n        input_specs, positions=positions))\ninput_specs = self.layer_norm(input_specs)\ninput_specs = self.dropout(input_specs)', note: 'Dropout p=0.1，仅训练时起作用。下方随机注意力实验按 eval 模式关闭 dropout，LayerNorm 使用初始 γ=1、β=0 的教学值。' },
      encoder: { title: '6 层跨电极 Transformer，默认 post-LN', shape: `[${count + 1},B,512] → [${count + 1},B,512]`, file: 'models/pt_model_custom.py', line: '#L104-L116', explanation: `${expandedLayer >= 0 ? `当前展开第 ${expandedLayer + 1} 层` : '每一层'}：8 个 head，每个 64 维。先 MHA，再残差相加 + LayerNorm；接着 FFN，再残差相加 + LayerNorm。结构相同，6 层参数独立。`, code: 'attn, weights = self.self_attn(x,x,x,\n    key_padding_mask=pad_mask)\nx = self.norm1(x + self.dropout1(attn))\nff = self.linear2(self.dropout(\n    self.activation(self.linear1(x))))\nx = self.norm2(x + self.dropout2(ff))', note: 'FFN 512→2048→512，activation=GELU，dropout=0.1。padding=True 的位置不作为 key；没有因果 mask。示意代码省略了输出注意力权重的设置。' },
      pretrain: { title: '预训练：CLS 判别配对，token 判别替换', shape: `CLS [B,512] → [B,1]；电极 [B,${count},512] → [B,${count},1]`, file: 'criterions/nsp_replace_only_pretrain.py', line: '#L26-L55', explanation: '一项 BCE 使用 CLS 的 logit 区分连续片段与随机配对；另一项 BCE 使用电极 token 的 logits 区分替换与保留。只有被选中的 token 才计入替换损失。', code: 'keep = (replace_label == 1) | (replace_label == 2)\nz = token_logits[keep]\ny = replace_label[keep]\ny[y == 2] = 0\nL_replace = BCEWithLogits(z, y)\nL = L_replace + BCEWithLogits(cls_z, labels)', note: '内部标签 1→替换→BCE y=1，2→保留→BCE y=0；0 忽略。spec_prediction_head 虽被 forward 计算，但该 criterion 不使用重建输出，不是 masked reconstruction loss。' },
      downstream: { title: '下游分类：第 0 个位置就是入口', shape: '[B,N+1,512] → [B,512] → [B,1]', file: 'models/pt_downstream_model.py', line: '#L12-L27', explanation: '加载预训练 PopT 后，取输出第 0 个 token（CLS），接新的 Linear(512,1)。无需把 N 个电极拼成固定长度。真实分类标签由任务数据集提供。', code: 'outs = self.upstream(\n    inputs, src_key_mask, positions,\n    intermediate_rep=True)\nh = outs[:,0,:]  # CLS\nlogits = self.linear_out(h) # 512→1\n# 默认下游 criterion：BCEWithLogitsLoss', note: 'frozen_upstream=False 更新 PopT 与新头；True 冻结 PopT、只训练新头。官方 criterion 也支持配置 mse，这里展示默认二分类 BCE 路线。' },
    };
    const item = items[id] || items.brainbert;
    set('source-title', item.title); set('source-shape', item.shape); set('source-explanation', item.explanation);
    set('source-code', item.code); set('source-note', item.note); set('source-file', item.file);
    get('source-link').href = REPO + item.file + item.line;
    each('[data-pt-module]', button => button.setAttribute('aria-pressed', String(button.dataset.ptModule === id)));
  }

  function renderExperiment() {
    const focused = document.activeElement;
    const focusQuery = focused?.hasAttribute('data-pt-query') ? focused.dataset.ptQuery : null;
    const focusCell = focused?.dataset.ptCell;
    attention = populationAttention(coordinates, head, splitSequence, usePosition);
    const row = attention.weights[query];
    const selectedCoords = query > 0 ? coordinates[query - 1] : [0, 0, 0];
    const maximum = Math.max(...attention.weights.flat());
    const points = coordinates.map(coords => [13 + coords[0] / 400 * 74, 13 + coords[1] / 400 * 74]);
    const queryPoint = query > 0 ? points[query - 1] : [50, 50];
    get('map').innerHTML = `<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${points.map((point, i) => `<line x1="${queryPoint[0]}" y1="${queryPoint[1]}" x2="${point[0]}" y2="${point[1]}" stroke="#087f69" stroke-width="${0.2 + row[i + 1] * 3}" opacity="${0.15 + row[i + 1] / Math.max(...row) * 0.75}"/>`).join('')}</svg>${points.map((point, i) => `<button type="button" class="pt-electrode" data-pt-query="${i + 1}" style="left:${point[0]}%;top:${point[1]}%" aria-label="选择电极 ${labelName(i + 1)} 作为 query" aria-pressed="${query === i + 1}">${labelName(i + 1)}</button>`).join('')}`;
    each('[data-pt-query]', button => button.setAttribute('aria-pressed', String(Number(button.dataset.ptQuery) === query)));
    set('coordinate-title', query ? `${labelName(query)} 的整数坐标（拖动改变 PE）` : 'CLS 使用 PE(0) × 4，不读取电极坐标');
    each('[data-pt-coordinate]', input => {
      const axis = Number(input.dataset.ptCoordinate);
      input.disabled = query === 0;
      input.value = selectedCoords[axis];
      input.closest('label').classList.toggle('pt-coordinate-disabled', query === 0);
      set(`coordinate-${axis}`, selectedCoords[axis]);
    });
    const names = Array.from({ length: count + 1 }, (_, i) => labelName(i));
    get('matrix').style.gridTemplateColumns = `28px repeat(${count + 1}, minmax(23px,1fr))`;
    get('matrix').innerHTML = `<span class="pt-matrix-label">Q↓ K→</span>${names.map(name => `<span class="pt-matrix-label">${name}</span>`).join('')}${attention.weights.map((weights, i) => `<span class="pt-matrix-label">${labelName(i)}</span>${weights.map((weight, j) => {
      const strength = Math.pow(weight / maximum, 0.7);
      const background = `rgb(${Math.round(241 - strength * 232)},${Math.round(248 - strength * 121)},${Math.round(237 - strength * 132)})`;
      return `<button type="button" class="pt-matrix-cell" data-pt-cell="${i},${j}" data-query="${i === query}" aria-pressed="${i === query && j === key}" aria-label="${labelName(i)} 读取 ${labelName(j)} 的权重 ${weight.toFixed(5)}" title="${labelName(i)} → ${labelName(j)}: ${weight.toFixed(5)}" style="background:${background};color:${strength > .62 ? '#fff' : '#174b3d'}">${weight.toFixed(2)}</button>`;
    }).join('')}`).join('')}`;
    set('read-pair', `${labelName(query)} → ${labelName(key)} 的权重`);
    set('read-weight', number(row[key]));
    set('row-sum', row.reduce((sum, value) => sum + value, 0).toFixed(6));
    set('read-score-label', 'softmax 前的 Q·K / √64');
    set('read-score', number(attention.scores[query][key]));
    const context = attention.contexts[query];
    set('context-first', `c[0] = ${number(context[0])}，c[1] = ${number(context[1])}`);
    const maxContext = Math.max(...context.slice(0, 16).map(Math.abs), 0.01);
    get('context-bars').innerHTML = context.slice(0, 16).map((value, i) => `<span class="pt-vector-bar" title="c[${i}]=${value.toFixed(5)}" style="height:${Math.max(3, Math.abs(value) / maxContext * 54)}px;background:${value < 0 ? '#d38a57' : '#087f69'};align-self:${value < 0 ? 'flex-end' : 'flex-start'}"></span>`).join('');
    set('pe-title', `${labelName(query)} 的位置编码 · ${usePosition ? '已加入输入' : '当前关闭，仅展示公式'}`);
    const sequenceId = query && splitSequence && query - 1 >= Math.floor(count / 2) ? 1 : 0;
    [...selectedCoords, sequenceId].forEach((value, i) => set(`pe-value-${i}`, `p = ${value}`));
    set('pe-first', `前 4 维：[${attention.positions[query].slice(0, 4).map(number).join(', ')}]。${query === 0 ? 'CLS 的四个 128 维段都使用位置 0。' : `该电极属于组 ${sequenceId}；编码拼接后与投影向量相加，再做 LayerNorm。`}`);
    if (focusCell) container.querySelector(`[data-pt-cell="${focusCell}"]`)?.focus({ preventScroll: true });
    else if (focusQuery !== null) container.querySelector(`[data-pt-query="${focusQuery}"]`)?.focus({ preventScroll: true });
  }

  function renderTasks() {
    const focusToken = document.activeElement?.dataset.ptToken;
    each('[data-pt-route]', button => button.setAttribute('aria-pressed', String(button.dataset.ptRoute === route)));
    get('pretrain-panel').classList.toggle('pt-hidden', route !== 'pretrain');
    get('downstream-panel').classList.toggle('pt-hidden', route !== 'downstream');
    each('[data-pt-pair]', button => button.setAttribute('aria-pressed', String(Number(button.dataset.ptPair) === pairLabel)));
    get('pair-b').dataset.random = String(pairLabel === 1);
    set('pair-time', pairLabel === 0 ? '下一连续片段 · seq_id 1' : '随机另一片段 · seq_id 1');
    set('pair-logit-value', pairLogit.toFixed(1));
    const pairLoss = binaryCrossEntropy(pairLogit, pairLabel);
    set('pair-result', `p(随机配对) = sigmoid(z) = ${number(sigmoid(pairLogit))}\ny = ${pairLabel} · L_ensemble = ${number(pairLoss)}`);
    get('token-picker').innerHTML = tokenLabels.map((label, i) => `<button type="button" data-pt-token="${i}" data-label="${label}" aria-label="检查 ${labelName(i + 1)}，当前标签${label}，${labelText(label)}" aria-pressed="${i === selectedToken}">${labelName(i + 1)} · ${labelText(label)}</button>`).join('');
    each('[data-pt-token-label]', button => button.setAttribute('aria-pressed', String(Number(button.dataset.ptTokenLabel) === tokenLabels[selectedToken])));
    const currentLabel = tokenLabels[selectedToken];
    set('token-label-description', `${labelName(selectedToken + 1)}：${currentLabel === 0 ? '内部标签 0 → loss mask=False；仍是注意力输入。' : currentLabel === 1 ? '内部标签 1 → 已替换 → BCE 目标 y=1。' : '内部标签 2 → 选中保留 → BCE 目标 y=0。'}`);
    get('token-logit').value = tokenLogits[selectedToken];
    set('token-logit-value', tokenLogits[selectedToken].toFixed(1));
    const replace = replacementLoss(tokenLogits, tokenLabels);
    get('token-table').innerHTML = tokenLabels.map((label, i) => {
      const eligible = label === 1 || label === 2;
      const target = label === 1 ? 1 : 0;
      return `<tr data-selected="${i === selectedToken}" class="${eligible ? '' : 'pt-excluded'}"><td>${labelName(i + 1)}</td><td>${label} · ${labelText(label)}</td><td>${eligible ? target : '—'}</td><td>${tokenLogits[i].toFixed(1)}</td><td>${eligible ? number(binaryCrossEntropy(tokenLogits[i], target)) : '不参与'}</td></tr>`;
    }).join('');
    set('replace-result', replace.count ? `选中 ${replace.count} / ${count} 个电极 token\nL_replace = 所选 BCE 均值 = ${number(replace.loss)}` : '没有选中 token；均值未定义。请至少选择一个替换或保留负例。');
    set('total-loss', replace.loss === null ? '等待有效 token' : number(pairLoss + replace.loss));
    each('[data-pt-freeze]', button => button.setAttribute('aria-pressed', String((button.dataset.ptFreeze === 'true') === frozen)));
    get('fine-backbone').dataset.train = String(!frozen);
    set('fine-backbone-state', frozen ? '冻结参数 · requires_grad=False' : '与分类头一起微调');
    set('gradient-path', frozen ? '梯度：BCE ← 新分类头；PopT 参数不更新。BrainBERT 特征已离线提取。' : '梯度：BCE ← 新分类头 ← PopT；BrainBERT 特征已离线提取。');
    each('[data-pt-fine-label]', button => button.setAttribute('aria-pressed', String(Number(button.dataset.ptFineLabel) === fineLabel)));
    set('fine-logit-value', fineLogit.toFixed(1));
    set('fine-result', `p(y=1) = sigmoid(${fineLogit.toFixed(1)}) = ${number(sigmoid(fineLogit))}\n任务标签 y = ${fineLabel}\nL_task = BCEWithLogits(z,y) = ${number(binaryCrossEntropy(fineLogit, fineLabel))}\n\nLoss 只来自下游分类；不叠加两项预训练 BCE。`);
    if (focusToken !== undefined) container.querySelector(`[data-pt-token="${focusToken}"]`)?.focus({ preventScroll: true });
  }

  on('click', event => {
    const button = event.target.closest('button');
    if (!button || !container.contains(button)) return;
    if (button.dataset.ptModule) showModule(button.dataset.ptModule);
    if (button.hasAttribute('data-pt-layer')) {
      const selected = Number(button.dataset.ptLayer);
      expandedLayer = expandedLayer === selected ? -1 : selected;
      each('[data-pt-layer]', item => item.setAttribute('aria-expanded', String(Number(item.dataset.ptLayer) === expandedLayer)));
      get('layer-detail').hidden = expandedLayer < 0;
      if (expandedLayer >= 0) { set('layer-caption', `第 ${expandedLayer + 1} 层 · 先残差相加，再 LayerNorm`); showModule('encoder'); }
    }
    if (button.hasAttribute('data-pt-query')) { query = Number(button.dataset.ptQuery); renderExperiment(); }
    if (button.dataset.ptCell) { [query, key] = button.dataset.ptCell.split(',').map(Number); renderExperiment(); }
    if (button.dataset.ptRoute) { route = button.dataset.ptRoute; renderTasks(); showModule(route === 'pretrain' ? 'pretrain' : 'downstream'); }
    if (button.hasAttribute('data-pt-pair')) { pairLabel = Number(button.dataset.ptPair); renderTasks(); }
    if (button.hasAttribute('data-pt-token')) { selectedToken = Number(button.dataset.ptToken); renderTasks(); }
    if (button.hasAttribute('data-pt-token-label')) { tokenLabels[selectedToken] = Number(button.dataset.ptTokenLabel); renderTasks(); }
    if (button.hasAttribute('data-pt-freeze')) { frozen = button.dataset.ptFreeze === 'true'; renderTasks(); }
    if (button.hasAttribute('data-pt-fine-label')) { fineLabel = Number(button.dataset.ptFineLabel); renderTasks(); }
  });
  on('input', event => {
    const input = event.target;
    if (input.hasAttribute('data-pt-coordinate') && query > 0) { coordinates[query - 1][Number(input.dataset.ptCoordinate)] = Number(input.value); renderExperiment(); }
    if (input.dataset.pt === 'pair-logit') { pairLogit = Number(input.value); renderTasks(); }
    if (input.dataset.pt === 'token-logit') { tokenLogits[selectedToken] = Number(input.value); renderTasks(); }
    if (input.dataset.pt === 'fine-logit') { fineLogit = Number(input.value); renderTasks(); }
  });
  on('change', event => {
    const input = event.target;
    if (input.dataset.pt === 'count') {
      count = Number(input.value); coordinates = defaultCoordinates(count); query = Math.min(query, count); key = Math.min(key, count); selectedToken = Math.min(selectedToken, count - 1);
      tokenLabels = Array.from({ length: count }, (_, i) => [1, 2, 0, 2][i % 4]);
      tokenLogits = Array.from({ length: count }, (_, i) => Math.round(Math.sin(i * 1.37 + 0.4) * 20) / 10);
      renderExperiment(); renderTasks(); showModule(selectedModule);
    }
    if (input.dataset.pt === 'head') { head = Number(input.value); renderExperiment(); }
    if (input.dataset.pt === 'split') { splitSequence = input.checked; renderExperiment(); }
    if (input.dataset.pt === 'position') { usePosition = input.checked; renderExperiment(); }
  });
  showModule(selectedModule);
  renderExperiment();
  renderTasks();
  return { render() { showModule(selectedModule); renderExperiment(); renderTasks(); }, destroy() { controller.abort(); container.replaceChildren(); } };
}
