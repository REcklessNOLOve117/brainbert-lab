let fineTuneInstances = 0;

/** Mount the sixth BrainBERT lesson. Returns a cleanup function. */
export function mountFineTune(container) {
  const uid = `ft-${++fineTuneInstances}`;
  let mode = 'frozen';
  let rank = 8;
  let logit = 0.8;
  let label = 1;
  let timer;
  let announceTimer;
  const number = value => Number(value).toLocaleString('en-US');
  const modeInfo = {
    frozen: { title: '冻结主干', subtitle: '仓库原生支持', count: () => 769,
      copy: 'BrainBERT 用 eval() + no_grad() 提取表示。分类头学习，输入投影与 Transformer 不产生梯度。',
      param: '只训练 768 个分类权重与 1 个 bias。',
      tag: '原生',
      flow: ['loss', 'classifier'] },
    full: { title: '全量微调', subtitle: '仓库原生支持', count: () => 42561025,
      copy: 'BCE 梯度从分类头传回 Transformer 和输入投影，调整表示来适应分类标签。重建预测头被跳过。',
      param: '有梯度：输入投影、6 层 Transformer、分类头；不计未参与分类的重建头。',
      tag: '原生',
      flow: ['loss', 'classifier', 'pool', 'transformer', 'projection'] },
    lora: { title: 'LoRA 微调', subtitle: '扩展教学方案', count: () => 18432 * rank + 769,
      copy: '固定原始权重 W，在每层 Q、V 投影旁加入可训练的低秩分支。梯度穿过计算图，但只更新 A、B 和分类头。',
      param: '6 层 × Q/V 两个投影 × r(768+768) + 分类头 769。',
      tag: '扩展示意',
      flow: ['loss', 'classifier', 'pool', 'transformer'] }
  };

  container.innerHTML = `
  <style>
  .ft-lab{--ft-ink:#172c40;--ft-muted:#60716f;--ft-green:#087f69;--ft-orange:#db7447;--ft-line:#e2e6df;color:var(--ft-ink);font-size:15px;line-height:1.65;max-width:1200px;margin:0 auto}
  .ft-lab *{box-sizing:border-box}.ft-lab button,.ft-lab input,.ft-lab select{font:inherit}.ft-lab button,.ft-lab select,.ft-lab input[type=range],.ft-lab input[type=radio]{cursor:pointer}
  .ft-lab button:focus-visible,.ft-lab input:focus-visible,.ft-lab select:focus-visible{outline:3px solid #db7447;outline-offset:4px}
  .ft-eyebrow{color:var(--ft-green);font-size:12px;letter-spacing:.14em;font-weight:750;text-transform:uppercase;margin:0 0 6px}.ft-lab h2{font-size:clamp(25px,3vw,37px);line-height:1.22;margin:0 0 13px;font-weight:750;letter-spacing:-.035em}.ft-lab h3{font-size:20px;line-height:1.35;margin:0 0 9px;letter-spacing:-.02em}.ft-lab p{margin:0 0 12px}.ft-muted{color:var(--ft-muted)}.ft-small{font-size:13px}.ft-mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:-.025em}.ft-intro{max-width:780px;margin-bottom:25px}.ft-kicker{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}.ft-badge{font-size:12px;line-height:1.4;border-radius:20px;padding:5px 10px;background:#ecf5f0;color:var(--ft-green);font-weight:650}.ft-badge--orange{background:#fbefe7;color:#a4522e}.ft-panel{background:#fff;border:1px solid var(--ft-line);border-radius:18px;padding:25px;margin-bottom:20px}.ft-panel--soft{background:#f7f7f1}.ft-row{display:flex;gap:16px;justify-content:space-between;align-items:flex-start;flex-wrap:wrap}.ft-heading{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.ft-heading h3{margin:0}.ft-mode-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:20px 0}.ft-mode{position:relative;display:block;border:1px solid var(--ft-line);border-radius:13px;padding:17px;background:#fff;cursor:pointer;transition:border-color .2s,background .2s}.ft-mode:hover{border-color:#8ab4a6}.ft-mode:has(input:checked){border:2px solid var(--ft-green);padding:16px;background:#f0f7f2}.ft-mode input{accent-color:var(--ft-green);margin:0 8px 0 0;width:16px;height:16px;vertical-align:middle}.ft-mode strong{font-size:16px}.ft-mode small{display:block;margin-left:28px;color:var(--ft-muted);font-size:12px;margin-top:3px}.ft-mode:has(input:focus-visible){outline:3px solid var(--ft-orange);outline-offset:3px}.ft-mode input:focus-visible{outline:none}.ft-mode-description{min-height:49px;margin-bottom:0!important}.ft-legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--ft-muted)}.ft-dot{display:inline-block;width:7px;height:7px;border-radius:100%;background:#bac7c1;margin-right:5px}.ft-dot--green{background:var(--ft-green)}.ft-dot--orange{background:var(--ft-orange)}.ft-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:18px;margin:24px 0 12px}.ft-node{position:relative;min-height:155px;background:#f7f8f5;border:1px solid var(--ft-line);border-radius:12px;padding:17px 12px;transition:background .25s,border-color .25s,box-shadow .25s}.ft-node:not(:last-child)::after{content:'→';position:absolute;right:-16px;top:62px;font-size:16px;color:#8fa096}.ft-node-title{display:block;font-size:14px;font-weight:750;margin-bottom:4px;line-height:1.4}.ft-node-shape{font-size:12px;color:var(--ft-muted)}.ft-node-state{display:block;font-size:11px;font-weight:650;margin-top:14px;color:#798b82}.ft-node[data-gradient=true]{background:#eef7f1;border-color:#c2ddce}.ft-node[data-gradient=true] .ft-node-state{color:var(--ft-green)}.ft-node[data-kind=loss]{background:#fcf3ec;border-color:#efdbcb}.ft-node[data-kind=loss] .ft-node-state{color:#b1653d}.ft-node[data-pulse=true]{animation:ft-gradient .7s ease both}.ft-note{font-size:13px;padding:12px 15px;border-left:3px solid #c7d5cb;background:#f6f7f1;border-radius:0 8px 8px 0;margin:15px 0}.ft-note--orange{border-left-color:var(--ft-orange);background:#fcf3ed}.ft-btn{border:0;border-radius:9px;background:var(--ft-green);color:#fff;min-height:42px;padding:9px 16px;font-weight:650;display:inline-flex;align-items:center;gap:8px}.ft-btn:hover{background:#066e5c}.ft-btn:disabled{opacity:.65;cursor:default}.ft-actions{display:flex;align-items:center;gap:13px;flex-wrap:wrap}.ft-live{font-size:13px;color:var(--ft-green);min-height:22px}.ft-two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.ft-two .ft-panel{margin-bottom:0}.ft-stat{font-size:clamp(30px,4vw,44px);font-weight:750;letter-spacing:-.05em;color:var(--ft-green);line-height:1.2;margin:10px 0 6px}.ft-stat-caption{font-size:13px;color:var(--ft-muted)}.ft-count-bar{height:8px;border-radius:8px;overflow:hidden;background:#e9eee6;margin:22px 0 10px}.ft-count-bar span{display:block;height:100%;background:var(--ft-green);min-width:3px;transition:width .3s}.ft-count-foot{display:flex;justify-content:space-between;font-size:12px;gap:12px;color:var(--ft-muted)}.ft-rank-row{margin:20px 0 15px;display:flex;align-items:center;justify-content:space-between;gap:12px}.ft-select{border:1px solid #d3ddd3;background:#fff;border-radius:8px;min-height:38px;padding:4px 12px;color:var(--ft-ink)}.ft-formula{background:#f6f7f2;border:1px solid #e8ece4;border-radius:9px;padding:13px 14px;font-size:14px;overflow-x:auto;white-space:nowrap}.ft-lora-box{border-top:1px solid var(--ft-line);margin-top:23px;padding-top:20px}.ft-lora-matrices{display:flex;justify-content:center;align-items:center;gap:11px;margin:17px 0}.ft-matrix{position:relative;border:1px solid #abcabd;background:repeating-linear-gradient(90deg,transparent,transparent 11px,#d8e6da 11px,#d8e6da 12px),repeating-linear-gradient(0deg,#eef5ed,#eef5ed 11px,#d8e6da 11px,#d8e6da 12px);height:65px;width:65px;display:flex;align-items:center;justify-content:center;border-radius:4px;color:var(--ft-green);font-size:18px;font-weight:750}.ft-matrix--a{width:65px;height:30px}.ft-matrix--b{width:30px;height:65px}.ft-matrix-label{font-size:11px;text-align:center;line-height:1.4;color:var(--ft-muted);margin-top:6px}.ft-matrix-op{font-size:20px;color:#8b9a8e}.ft-slider-row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:20px;margin-bottom:8px}.ft-value{font-size:18px;font-weight:750;color:var(--ft-green)}.ft-range{width:100%;accent-color:var(--ft-green);min-height:24px;margin:0}.ft-range-ends{display:flex;justify-content:space-between;font-size:11px;color:#87948d}.ft-label-choice{display:flex;gap:12px;border:0;padding:0;margin:16px 0}.ft-label-choice legend{font-size:13px;font-weight:650;margin-bottom:9px}.ft-label{border:1px solid var(--ft-line);border-radius:8px;padding:7px 12px;font-size:13px}.ft-label input{accent-color:var(--ft-green);margin-right:5px}.ft-label:has(input:checked){background:#f0f7f1;border-color:#aacdbb}.ft-metrics{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:17px 0}.ft-metric{border:1px solid var(--ft-line);padding:14px;border-radius:11px}.ft-metric dt{font-size:12px;color:var(--ft-muted)}.ft-metric dd{margin:5px 0 0;font-size:26px;font-weight:750;line-height:1.2;letter-spacing:-.025em}.ft-metric--loss dd{color:#b76237}.ft-prob-track{position:relative;background:#ecf1e9;height:14px;border-radius:10px;overflow:hidden;margin-bottom:9px}.ft-prob-track span{height:100%;background:var(--ft-green);display:block;transition:width .1s}.ft-prob-caption{display:flex;justify-content:space-between;font-size:11px;color:var(--ft-muted)}.ft-gradient-equation{font-size:13px;border-top:1px solid var(--ft-line);padding-top:15px;margin-top:15px}.ft-footer{display:flex;gap:11px;align-items:flex-start;margin:22px 0 5px;font-size:13px;color:var(--ft-muted)}.ft-footer-symbol{color:var(--ft-green);font-size:18px;line-height:1.4}.ft-sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
  @keyframes ft-gradient{0%{box-shadow:0 0 0 0 #db744700}45%{box-shadow:0 0 0 5px #db744744;background:#fff0df;border-color:#db7447}100%{box-shadow:0 0 0 0 #db744700}}
  @media(max-width:850px){.ft-flow{gap:13px}.ft-node{padding:14px 9px}.ft-node:not(:last-child)::after{right:-12px}.ft-two{grid-template-columns:1fr}.ft-panel{padding:21px}.ft-node-title{font-size:13px}}
  @media(max-width:580px){.ft-mode-grid{grid-template-columns:1fr;gap:8px}.ft-mode{padding:12px 14px}.ft-mode:has(input:checked){padding:11px 13px}.ft-mode small{display:inline;margin-left:7px}.ft-mode-description{min-height:0}.ft-flow{grid-template-columns:1fr;margin-top:18px;gap:21px}.ft-node{min-height:82px;padding:12px 15px}.ft-node:not(:last-child)::after{content:'↓';right:calc(50% - 5px);top:auto;bottom:-23px}.ft-node-state{position:absolute;right:14px;top:13px;margin-top:0}.ft-node-title{max-width:65%;font-size:14px}.ft-panel{padding:18px;border-radius:14px}.ft-lora-matrices{gap:8px}.ft-mode-grid{margin:16px 0}.ft-count-foot{font-size:11px}}
  @media(prefers-reduced-motion:reduce){.ft-lab *{animation-duration:.01ms!important;transition:none!important}}
  </style>
  <section class="ft-lab" aria-labelledby="${uid}-heading">
    <div class="ft-intro">
      <div class="ft-kicker"><span class="ft-eyebrow">07 / Fine-tuning</span><span class="ft-badge">从表示到分类</span></div>
      <h2 id="${uid}-heading">让脑电表示，学会回答一个问题。</h2>
      <p class="ft-muted">预训练学会重建，下游任务用标签训练分类。这里沿用仓库的<strong>中心 10 帧均值池化 → 二分类头</strong>，观察训练方式如何改变梯度路径。</p>
    </div>
    <div class="ft-panel">
      <div class="ft-row"><div><p class="ft-eyebrow">01 / Choose what learns</p><h3>哪些参数参与学习？</h3></div><div class="ft-legend"><span><i class="ft-dot ft-dot--green"></i>可训练参数</span><span><i class="ft-dot"></i>固定参数 / 无参数</span><span><i class="ft-dot ft-dot--orange"></i>反向传播</span></div></div>
      <div class="ft-mode-grid" role="radiogroup" aria-label="选择微调方式">
        ${Object.entries(modeInfo).map(([key, info]) => `<label class="ft-mode"><input type="radio" name="${uid}-mode" value="${key}" ${key === mode ? 'checked' : ''}><strong>${info.title}</strong><small>${info.subtitle}</small></label>`).join('')}
      </div>
      <p class="ft-mode-description" data-ft="mode-description"></p>
      <div class="ft-flow" aria-label="分类计算图">
        <div class="ft-node" data-kind="projection"><strong class="ft-node-title">输入投影</strong><span class="ft-node-shape ft-mono">40 → 768 + PE/LN</span><span class="ft-node-state" data-state="projection"></span></div>
        <div class="ft-node" data-kind="transformer"><strong class="ft-node-title">Transformer</strong><span class="ft-node-shape ft-mono">6 层 · [B,T,768]</span><span class="ft-node-state" data-state="transformer"></span></div>
        <div class="ft-node" data-kind="pool"><strong class="ft-node-title">中心 10 帧均值</strong><span class="ft-node-shape ft-mono">[B,T,768] → [B,768]</span><span class="ft-node-state" data-state="pool"></span></div>
        <div class="ft-node" data-kind="classifier"><strong class="ft-node-title">分类头</strong><span class="ft-node-shape ft-mono">Linear(768, 1)</span><span class="ft-node-state" data-state="classifier"></span></div>
        <div class="ft-node" data-kind="loss"><strong class="ft-node-title">BCE 分类损失</strong><span class="ft-node-shape ft-mono">logit z + 标签 y</span><span class="ft-node-state">反向传播起点</span></div>
      </div>
      <div class="ft-note">每个样本先得到一个 logit。训练用 <span class="ft-mono">BCEWithLogitsLoss</span>，推理再用 sigmoid 转成概率。分类路线跳过重建预测头。</div>
      <div class="ft-actions"><button class="ft-btn" type="button" data-ft="backprop"><span aria-hidden="true">←</span> 运行一次反向传播</button><span class="ft-live" data-ft="backprop-status" role="status" aria-live="polite">观察 loss 怎样传回可训练模块。</span></div>
    </div>
    <div class="ft-two">
      <div class="ft-panel">
        <p class="ft-eyebrow">02 / Parameter budget</p><h3>更新多少参数？</h3>
        <p class="ft-stat-caption" data-ft="count-title">当前可训练参数</p><div class="ft-stat ft-mono" data-ft="count"></div>
        <p class="ft-small ft-muted" data-ft="parameter-note"></p>
        <div class="ft-count-bar" aria-hidden="true"><span data-ft="count-bar"></span></div><div class="ft-count-foot"><span data-ft="fraction"></span><span>全量分类：42.56M</span></div>
        <div class="ft-rank-row"><label for="${uid}-rank"><strong>LoRA rank</strong><span class="ft-small ft-muted"> · 低秩维度 r</span></label><select class="ft-select" id="${uid}-rank" data-ft="rank" aria-label="LoRA rank">${[4,8,16,32,64].map(r=>`<option value="${r}" ${r===8?'selected':''}>r = ${r}</option>`).join('')}</select></div>
        <div class="ft-formula ft-mono" data-ft="parameter-formula"></div>
        <div class="ft-lora-box">
          <div class="ft-heading"><h3>LoRA：给 W 加一条小分支</h3><span class="ft-badge ft-badge--orange">扩展示意</span></div>
          <div class="ft-lora-matrices" aria-label="LoRA 的原始矩阵 W 加低秩矩阵乘积 BA">
            <div><div class="ft-matrix">W</div><div class="ft-matrix-label">768 × 768<br>固定</div></div><span class="ft-matrix-op">+</span><span class="ft-mono ft-small">α/r</span><span class="ft-matrix-op">×</span><div><div class="ft-matrix ft-matrix--b">B</div><div class="ft-matrix-label" data-ft="b-shape"></div></div><span class="ft-matrix-op">×</span><div><div class="ft-matrix ft-matrix--a">A</div><div class="ft-matrix-label" data-ft="a-shape"></div></div>
          </div>
          <div class="ft-formula ft-mono">W′ = W + (α/r) BA</div><p class="ft-small ft-muted" style="margin:11px 0 0">A、B 的乘积与 W 同形状；α 控制分支缩放。rank 改变新增参数量，不能据此推断准确率。</p>
          <div class="ft-note ft-note--orange">原仓库<strong>未实现 LoRA</strong>。此处按每层 Q/V 两个投影计算；源码采用融合 QKV 参数，实际接入需适配。</div>
        </div>
      </div>
      <div class="ft-panel">
        <div class="ft-heading"><div><p class="ft-eyebrow">03 / Classification loss</p><h3>让 logit 对齐标签</h3></div><span class="ft-badge ft-badge--orange">教学数值</span></div>
        <p class="ft-small ft-muted" style="margin-top:13px">拖动一个手动设定的 logit，观察 sigmoid 和 BCE。这里展示公式关系，数值不是模型预测。</p>
        <div class="ft-slider-row"><label for="${uid}-logit"><strong>分类 logit · z</strong></label><output class="ft-value ft-mono" data-ft="logit-output" for="${uid}-logit"></output></div>
        <input class="ft-range" id="${uid}-logit" data-ft="logit" type="range" min="-5" max="5" step="0.1" value="0.8" aria-label="手动设定分类 logit"><div class="ft-range-ends"><span>−5 · 倾向类别 0</span><span>+5 · 倾向类别 1</span></div>
        <fieldset class="ft-label-choice"><legend>真实标签 y</legend><label class="ft-label"><input type="radio" name="${uid}-label" value="0">0 · 负类</label><label class="ft-label"><input type="radio" name="${uid}-label" value="1" checked>1 · 正类</label></fieldset>
        <dl class="ft-metrics"><div class="ft-metric"><dt>σ(z) · 正类概率</dt><dd class="ft-mono" data-ft="probability"></dd></div><div class="ft-metric ft-metric--loss"><dt>单样本 BCE</dt><dd class="ft-mono" data-ft="bce"></dd></div></dl>
        <div class="ft-prob-track" aria-hidden="true"><span data-ft="probability-bar"></span></div><div class="ft-prob-caption"><span>0%</span><span>50%</span><span>100%</span></div>
        <div class="ft-formula ft-mono" style="margin-top:22px">p = 1 / (1 + exp(−z))</div><div class="ft-formula ft-mono" style="margin-top:8px">L = −[y ln p + (1−y) ln(1−p)]</div>
        <p class="ft-gradient-equation">传回分类头的梯度：<span class="ft-mono">∂L/∂z = p − y = <strong data-ft="derivative"></strong></span></p>
        <p class="ft-small ft-muted" data-ft="loss-explanation"></p>
        <div class="ft-note">仓库的分类任务是二分类。中心 10 帧池化没有参数，负责把多个时间帧汇总为 768 维向量。</div>
        <p class="ft-small ft-muted" style="margin-bottom:0">图中采用固定长度样本。源码按 batch 序列中点取 10 帧，没有按每条有效长度重新计算中心。</p>
      </div>
    </div>
    <div class="ft-footer"><span class="ft-footer-symbol" aria-hidden="true">↗</span><span>依据 <span class="ft-mono">finetune_model.py</span> 与 <span class="ft-mono">finetune_criterion.py</span>。参数数目按 large 架构推算；本页没有加载权重或执行训练。</span></div>
    <span class="ft-sr-only" data-ft="metric-announcement" aria-live="polite"></span>
  </section>`;

  const get = name => container.querySelector(`[data-ft="${name}"]`);
  const nodes = [...container.querySelectorAll('.ft-node')];
  const controller = new AbortController();
  const on = (element, event, handler) => element.addEventListener(event, handler, { signal: controller.signal });

  function updateMode() {
    clearTimeout(timer);
    get('backprop').disabled = false;
    nodes.forEach(node => node.removeAttribute('data-pulse'));
    const info = modeInfo[mode];
    const count = info.count();
    get('mode-description').textContent = info.copy;
    get('count-title').textContent = `${info.title} · 可训练参数`;
    get('count').textContent = number(count);
    get('parameter-note').textContent = info.param;
    const percent = count / 42561025 * 100;
    get('fraction').textContent = `占全量 ${percent < 0.01 ? percent.toFixed(4) : percent.toFixed(2)}%`;
    get('count-bar').style.width = `${percent}%`;
    get('parameter-formula').textContent = mode === 'lora' ? `18,432 × ${rank} + 769 = ${number(count)}` : mode === 'full' ? '42,560,256 主干 + 769 分类头' : '768 权重 + 1 bias = 769';
    get('rank').disabled = mode !== 'lora';
    get('rank').title = mode !== 'lora' ? '选择 LoRA 微调后可调整 rank' : '改变每层 Q/V 分支的低秩维度';
    get('b-shape').innerHTML = `768 × ${rank}<br>可训练`;
    get('a-shape').innerHTML = `${rank} × 768<br>可训练`;
    container.querySelector('[data-state="projection"]').textContent = mode === 'full' ? '更新全部参数' : mode === 'frozen' ? '固定 · no_grad' : '固定参数';
    container.querySelector('[data-state="transformer"]').textContent = mode === 'full' ? '更新全部参数' : mode === 'lora' ? '仅更新 Q/V 的 A、B' : '固定 · no_grad';
    container.querySelector('[data-state="pool"]').textContent = '无参数 · mean';
    container.querySelector('[data-state="classifier"]').textContent = '更新 769 个参数';
    nodes.forEach(node => {
      const kind = node.dataset.kind;
      node.dataset.gradient = String(kind === 'classifier' || kind === 'projection' && mode === 'full' || kind === 'transformer' && mode !== 'frozen');
    });
    get('backprop-status').textContent = mode === 'frozen' ? '梯度止于分类头，主干表示已分离。' : mode === 'lora' ? '梯度通过主干，原始 W 固定，仅更新适配器与分类头。' : '分类梯度将传回输入投影与 Transformer。';
  }

  function updateLoss(announce = false) {
    const probability = 1 / (1 + Math.exp(-logit));
    const loss = Math.max(logit, 0) - label * logit + Math.log1p(Math.exp(-Math.abs(logit)));
    const derivative = probability - label;
    get('logit-output').textContent = `${logit > 0 ? '+' : ''}${logit.toFixed(1)}`;
    get('probability').textContent = `${(probability * 100).toFixed(1)}%`;
    get('probability-bar').style.width = `${probability * 100}%`;
    get('bce').textContent = loss.toFixed(4);
    get('derivative').textContent = `${derivative > 0 ? '+' : ''}${derivative.toFixed(4)}`;
    get('loss-explanation').textContent = label === 1 ? '标签为 1：正类概率越高，损失越小；负梯度会推动 logit 增大。' : '标签为 0：正类概率越低，损失越小；正梯度会推动 logit 减小。';
    if (announce) {
      clearTimeout(announceTimer);
      announceTimer = setTimeout(() => { get('metric-announcement').textContent = `Logit ${logit.toFixed(1)}，标签 ${label}，正类概率 ${(probability * 100).toFixed(1)}%，BCE ${loss.toFixed(4)}`; }, 250);
    }
  }

  container.querySelectorAll(`input[name="${uid}-mode"]`).forEach(input => on(input, 'change', () => { mode = input.value; updateMode(); }));
  container.querySelectorAll(`input[name="${uid}-label"]`).forEach(input => on(input, 'change', () => { label = Number(input.value); updateLoss(true); }));
  on(get('rank'), 'change', event => { rank = Number(event.target.value); updateMode(); });
  on(get('logit'), 'input', event => { logit = Number(event.target.value); updateLoss(true); });
  on(get('backprop'), 'click', () => {
    const flow = [...modeInfo[mode].flow];
    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    get('backprop').disabled = true;
    nodes.forEach(node => node.removeAttribute('data-pulse'));
    const descriptions = { loss: 'BCE 对 logit 求导：p − y。', classifier: '计算分类头权重与 bias 的梯度。', pool: 'mean 将梯度分配给中心 10 帧。', transformer: mode === 'lora' ? '梯度穿过 Transformer，仅 A、B 可更新。' : '计算 6 层 Transformer 参数的梯度。', projection: '梯度到达输入投影和 LayerNorm。' };
    let index = 0;
    const next = () => {
      if (index === flow.length) {
        get('backprop').disabled = false;
        get('backprop-status').textContent = mode === 'frozen' ? '完成：仅分类头有梯度。' : mode === 'lora' ? '完成：Q/V 适配器与分类头有梯度，原始 W 固定。' : '完成：分类头、Transformer、输入投影有梯度。';
        return;
      }
      const kind = flow[index++];
      container.querySelector(`[data-kind="${kind}"]`).dataset.pulse = 'true';
      get('backprop-status').textContent = descriptions[kind];
      timer = setTimeout(next, reducedMotion ? 30 : 620);
    };
    next();
  });
  updateMode();
  updateLoss();
  return () => { clearTimeout(timer); clearTimeout(announceTimer); controller.abort(); container.replaceChildren(); };
}
