import { makeInputEmbeddings, computeAttention, makeMaskedExample, stftWaveform } from './math.js';
import { mountFineTune } from './finetune.js';
import { mountArchitecture } from './architecture.js';

const $ = (id) => document.getElementById(id);
const state = { page: 0, frame: 93, band: 12, key: 83, head: 0, temperature: 1, padding: 0, source: 'real', version: 0, projection: 'norm', maskMode: 'both', probability: .1, seed: 7, error: .5, alpha: 2, output: 'embedding', spectrumView: 'zscore' };
const pages = [
  ['architecture','MODEL ARCHITECTURE','先把整张模型地图，放在眼前。','从输入编码到六层 Encoder，再到表示、重建与分类。点击模块，查看它的形状、用途和源码。','先看清模型的完整结构'],
  ['stft','SIGNAL → SPECTROGRAM','先看见，一段脑电里有什么。','同一段 5 秒信号，从时间与频率两个角度观察。拖动时间帧，把两张图对起来。','先对齐时间与频率'],
  ['tokens','SPECTROGRAM → TOKENS','一个时间帧，就是一个 token。','取出时频图的一列，跟着它从 40 维频率特征，走向 768 维输入表示。','理解 token 的内容'],
  ['attention','TOKENS → CONTEXT','每一帧，都可以向其他帧提问。','选择 query 时间帧，切换 head，检查 Q/K/V 怎样决定信息的读取与汇总。','沿时间建立上下文'],
  ['masking','RECONSTRUCTION → LEARNING','遮住一部分，学会理解其余部分。','检查哪些位置参与监督。调整模拟误差，亲手核对一次重建 loss。','让 loss 对应到单元格'],
  ['outputs','BACKBONE → REPRESENTATION','同一个主干，选择不同的出口。','embedding 在预测头之前。切换任务，看清输出形状、池化位置和返回代码。','区分表示与预测'],
  ['finetuning','REPRESENTATION → CLASSIFICATION','把学到的表示，用在你的任务里。','加载预训练主干，接上分类头。选择冻结、全量微调或 LoRA，沿梯度路径看哪些参数会改变。','理解分类与参数更新']
];
let original, data, playTimer = null, embCache = new Map(), attentionCache = new Map(), maskCache = null, frameRAF = 0;
const mono = 'ui-monospace, SFMono-Regular, Consolas, monospace';
const colors = { ink:'#567569', muted:'#71816e', line:'#eaf0e4', green:'#087f69', light:'#c7e0a0', orange:'#d39465' };
const clamp = (x,a,b) => Math.max(a, Math.min(b,x));
const num = (x,n=3) => Number.isFinite(x) ? x.toFixed(n) : '—';
const pad = (x) => String(x).padStart(3,'0');
const escape = (x) => String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');

function context(id, overrideHeight) {
  const canvas = typeof id === 'string' ? $(id) : id;
  const w = Math.max(100, canvas.getBoundingClientRect().width || 600);
  const h = overrideHeight || Number(canvas.dataset.height) || 200;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr); canvas.style.height = h+'px';
  const c = canvas.getContext('2d'); c.setTransform(dpr,0,0,dpr,0,0); c.clearRect(0,0,w,h);
  c.font='9px '+mono; c.lineWidth=1;
  return { c, w, h, canvas };
}
const paletteStops = [[0,[26,60,70]],[.32,[71,125,124]],[.58,[153,183,148]],[.8,[220,234,163]],[1,[224,165,92]]];
function palette(x) {
  x=clamp(x,0,1); let i=1; while(i<paletteStops.length-1 && x>paletteStops[i][0]) i++;
  const a=paletteStops[i-1],b=paletteStops[i],t=(x-a[0])/(b[0]-a[0]); return a[1].map((v,j)=>Math.round(v+(b[1][j]-v)*t));
}
function paintHeat(id, values, opts={}) {
  const {c,w,h,canvas}=context(id); const cols=values.length, rows=values[0].length;
  const p={x:45,y:9,w:w-60,h:h-40,cols,rows,kind:opts.kind||'heat'}; canvas._plot=p;
  const off=document.createElement('canvas'); off.width=cols;off.height=rows; const oc=off.getContext('2d'), img=oc.createImageData(cols,rows);
  for(let t=0;t<cols;t++) for(let f=0;f<rows;f++) {
    let rgb=palette(opts.magnitude ? Math.log1p(Math.max(0,values[t][f])/opts.magnitudeMax*30)/Math.log(31) : (values[t][f]+2.5)/7);
    if(opts.error) {
      const a=clamp(values[t][f]/Math.max(.1,state.error*1.2),0,1);
      rgb=opts.mask && !opts.mask[t][f] ? [243,246,238] : [249-45*a,246-110*a,232-120*a].map(Math.round);
    }
    const k=((rows-1-f)*cols+t)*4;img.data.set([...rgb,255],k);
  }
  oc.putImageData(img,0,0);c.imageSmoothingEnabled=false;c.drawImage(off,p.x,p.y,p.w,p.h);
  if(opts.supervision){c.strokeStyle='#f0f6ba9c';c.lineWidth=.55;for(let t=0;t<cols;t++)for(let f=0;f<rows;f++)if(opts.supervision[t][f])c.strokeRect(p.x+t/cols*p.w,p.y+(rows-1-f)/rows*p.h,p.w/cols,p.h/rows);}

  c.fillStyle=colors.muted;c.textAlign='right';
  const ticks=rows===40?[0,10,20,30,39]:[0,12,24,36,rows-1];
  for(const f of ticks){const y=p.y+p.h-(f+.5)/rows*p.h;c.fillText(opts.dimensions?String(f+1):num(data.stft.frequency[f],0),p.x-8,y+3);}
  c.textAlign='center';const ts=[0,Math.floor(cols*.25),Math.floor(cols*.5),Math.floor(cols*.75),cols-1];
  for(const t of ts)c.fillText(num(data.stft.time[t],2)+' s',p.x+(t+.5)/cols*p.w,p.y+p.h+18);
  c.textAlign='left';c.font='8px '+mono;c.fillText(opts.dimensions?'dim':'Hz',3,10);
  if(opts.selection!==false){const x=p.x+(state.frame+.5)/cols*p.w;c.fillStyle='#e7f7cd24';c.fillRect(x-p.w/cols/2,p.y,Math.max(1,p.w/cols),p.h);c.strokeStyle='#eefacb';c.lineWidth=1.2;c.beginPath();c.moveTo(x,p.y);c.lineTo(x,p.y+p.h);c.stroke();
    if(!opts.dimensions){const y=p.y+p.h-(state.band+.5)/rows*p.h;c.strokeStyle='#ffffffb0';c.strokeRect(p.x+state.frame/cols*p.w-2,y-p.h/rows/2-2,p.w/cols+4,p.h/rows+4);}}
}
function drawWave() {
  const {c,w,h,canvas}=context('waveform');const p={x:45,y:10,w:w-60,h:h-41,kind:'wave'};canvas._plot=p;
  const wav=data.sample.waveform, lo=Math.min(...wav),hi=Math.max(...wav),range=(hi-lo)||1;
  const sy=v=>p.y+p.h-(v-lo+range*.08)/(range*1.16)*p.h;
  c.strokeStyle=colors.line;c.fillStyle=colors.muted;c.textAlign='right';
  for(let j=0;j<3;j++){const v=lo+(hi-lo)*j/2,y=sy(v);c.beginPath();c.moveTo(p.x,y);c.lineTo(p.x+p.w,y);c.stroke();c.fillText(num(v,0),p.x-8,y+3);}
  const t=data.stft.time[state.frame], left=(t-200/2048)/5,right=(t+200/2048)/5;
  c.fillStyle='#bddda85c';c.fillRect(p.x+left*p.w,p.y,(right-left)*p.w,p.h);
  c.strokeStyle='#608c86';c.lineWidth=.8;c.beginPath();
  const buckets=Math.ceil(p.w),step=wav.length/buckets;
  for(let j=0;j<buckets;j++){let min=Infinity,max=-Infinity;for(let k=Math.floor(j*step);k<Math.min(wav.length,Math.floor((j+1)*step));k++){min=Math.min(min,wav[k]);max=Math.max(max,wav[k]);}const x=p.x+j/buckets*p.w;c.moveTo(x,sy(min));c.lineTo(x,sy(max));}c.stroke();
  const x=p.x+t/5*p.w;c.strokeStyle=colors.green;c.lineWidth=1;c.beginPath();c.moveTo(x,p.y);c.lineTo(x,p.y+p.h);c.stroke();
  c.fillStyle=colors.muted;c.textAlign='center';for(let i=0;i<=5;i++)c.fillText(i+' s',p.x+i/5*p.w,p.y+p.h+19);
}
function drawBars(id,values,opts={}) {
  const {c,w,h,canvas}=context(id),p={x:35,y:15,w:w-48,h:h-42,cols:values.length,kind:'bars'};canvas._plot=p;
  const max=Math.max(2.5,...values.map(Math.abs))*1.12,min=opts.nonnegative?0:-max,range=max-min;
  const sy=v=>p.y+p.h-(v-min)/range*p.h,zero=sy(0);
  c.strokeStyle=colors.line;c.fillStyle=colors.muted;c.textAlign='right';
  for(const v of [min,0,max]){const y=sy(v);c.beginPath();c.moveTo(p.x,y);c.lineTo(p.x+p.w,y);c.stroke();c.fillText(num(v,opts.precision??1),p.x-6,y+3);}
  for(let i=0;i<values.length;i++){const v=values[i],x=p.x+i/values.length*p.w,bw=p.w/values.length;c.fillStyle=(i===state.band&&!opts.noSelected)?colors.orange:colors.green;c.globalAlpha=(i===state.band&&!opts.noSelected)?1:.65;c.fillRect(x+.7,Math.min(zero,sy(v)),Math.max(.8,bw-1.4),Math.max(.7,Math.abs(zero-sy(v))));}c.globalAlpha=1;
  c.fillStyle=colors.muted;c.textAlign='center';const ticks=[0,Math.floor(values.length/2),values.length-1];for(const i of ticks)c.fillText(opts.frequency?num(data.stft.frequency[i],0)+' Hz':String(i+1),p.x+(i+.5)/values.length*p.w,p.y+p.h+18);
}
function drawGrid(id,vector) {
  const {c,w,h,canvas}=context(id);const cols=48,rows=Math.ceil(vector.length/cols),gap=2,cw=(w-12)/cols,ch=(h-26)/rows;
  for(let i=0;i<vector.length;i++){const rgb=palette((vector[i]+2.5)/6);c.fillStyle='rgb('+rgb.join(',')+')';c.fillRect(6+(i%cols)*cw,3+Math.floor(i/cols)*ch,Math.max(1,cw-gap),Math.max(1,ch-gap));}
  c.fillStyle=colors.muted;c.font='8px '+mono;c.fillText('768 dimensions · 48 × 16',6,h-2);c.textAlign='right';c.fillText('负值 ← 0 → 正值',w-5,h-2);canvas._plot={kind:'grid'};
}
function getEmbeddings(stage='norm') {if(!embCache.has(stage))embCache.set(stage,makeInputEmbeddings(data.stft.values,stage));return embCache.get(stage);}
function getAttention() {const key=[state.head,state.temperature,state.padding].join(':');if(!attentionCache.has(key))attentionCache.set(key,computeAttention(getEmbeddings(),state.head,{temperature:state.temperature,padding:state.padding}));return attentionCache.get(key);}
function syncFrame() {
  $('frame-slider').value=state.frame;$('frame-number').textContent=pad(state.frame+1);$('frame-time').textContent=num(data.stft.time[state.frame],3)+' s';
  $('selected-cell').textContent='f = '+num(data.stft.frequency[state.band],2)+' Hz';$('spectrum-frame').textContent='t'+pad(state.frame+1);
  $('band-readout').textContent=num(data.stft.frequency[state.band],2)+' Hz';$('feature-readout').textContent=num(data.stft.values[state.frame][state.band]);
}
function renderSTFT() {
  const magnitude=state.spectrumView==='magnitude',values=magnitude?data.stft.magnitude:data.stft.values;
  const magnitudeMax=state.source==='synthetic'?2:Math.max(...original.stft.magnitude.flat());
  drawWave();paintHeat('spectrogram',values,{magnitude,magnitudeMax});drawBars('spectrum',values[state.frame],{frequency:true,nonnegative:magnitude,precision:magnitude?2:1});
  $('heat-scale').textContent=magnitude?'abs(STFT)':'z-score';$('feature-readout').textContent=num(values[state.frame][state.band]);$('feature-unit').textContent=magnitude?'STFT 幅值':'标准化幅值';
  $('spectrum-note').textContent=magnitude?'当前显示标准化前的幅值。逐频点 z-score 会消去绝对幅值差异；模型输入采用标准化特征。':'横轴是频率。每一根柱子，都是这个时间帧的一项输入特征。';
}
function renderTokens() {
  paintHeat('token-heatmap',data.stft.values);drawBars('input-vector',data.stft.values[state.frame],{frequency:true});drawGrid('embedding-grid',getEmbeddings(state.projection)[state.frame]);
  $('token-strip').querySelectorAll('button').forEach((b,i)=>{b.classList.toggle('selected',i===state.frame);b.setAttribute('aria-pressed',String(i===state.frame));});
  const desc={linear:'eₜ = W xₜ + b。演示 W 的形状为 [768, 40]，所有时间帧共享同一组参数。',position:'加入固定的正弦 / 余弦位置编码，时间位置不同，即使频率特征相同也能产生不同表示。',norm:'LayerNorm：调整当前 token 的 768 个维度；模型推理时 dropout 关闭。'};
  $('projection-description').textContent=desc[state.projection];
}
function renderAttention() {
  const a=getAttention(),T=a.weights.length;const {c,w,h,canvas}=context('attention-map');const p={x:45,y:12,w:w-63,h:h-46,cols:T,rows:T,kind:'attention'};canvas._plot=p;
  const off=document.createElement('canvas');off.width=T;off.height=T;const oc=off.getContext('2d'),im=oc.createImageData(T,T);let max=0;
  for(const row of a.weights)for(const v of row)max=Math.max(max,v);
  for(let q=0;q<T;q++)for(let k=0;k<T;k++){const rgb=k>=T-state.padding?[214,222,209]:palette(Math.sqrt(a.weights[q][k]/max));im.data.set([...rgb,255],(q*T+k)*4);}oc.putImageData(im,0,0);c.imageSmoothingEnabled=false;c.drawImage(off,p.x,p.y,p.w,p.h);
  c.strokeStyle='#ddf6ae';c.lineWidth=1;c.strokeRect(p.x,p.y+state.frame/T*p.h,p.w,Math.max(1,p.h/T));c.strokeStyle='#f0b07b';c.strokeRect(p.x+state.key/T*p.w,p.y,Math.max(1,p.w/T),p.h);
  c.fillStyle=colors.muted;c.textAlign='center';for(const i of [0,46,93,139,185])c.fillText(String(i+1),p.x+(i+.5)/T*p.w,p.y+p.h+18);c.textAlign='right';for(const i of [0,46,93,139,185])c.fillText(String(i+1),p.x-8,p.y+(i+.5)/T*p.h+3);c.textAlign='left';c.font='8px '+mono;c.fillText('query',1,8);c.textAlign='right';c.fillText('key →',w-6,h-2);
  $('attention-query-title').textContent='时间帧 '+pad(state.frame+1)+' 在关注哪里';
  const row=a.weights[state.frame];drawWeightRow(row);
  const top=Array.from(row,(v,i)=>({v,i})).sort((x,y)=>y.v-x.v).slice(0,4);
  $('attention-top').innerHTML=top.map((x,n)=>'<button data-key="'+x.i+'" aria-label="查看 key 帧 '+(x.i+1)+'"><span>'+String(n+1).padStart(2,'0')+' / key '+pad(x.i+1)+'</span><b>'+num(x.v*100,2)+'%</b></button>').join('');
  const weighted=new Float64Array(64);for(let k=0;k<T;k++)for(let d=0;d<64;d++)weighted[d]+=row[k]*a.V[k][d];
  const items=[['Q','当前 query',a.Q[state.frame],'帧 '+pad(state.frame+1)+' · 发起读取'],['K','选中的 key',a.K[state.key],'帧 '+pad(state.key+1)+' · 匹配 Q'],['V','选中的 value',a.V[state.key],'与该 key 对应的内容'],['A · V','加权汇总',weighted,'当前 query 的 64 维 head 输出']];
  $('qkv-cards').innerHTML=items.map(x=>'<div class="card"><span class="eyebrow">'+x[0]+' / [64]</span><h3>'+x[1]+'</h3><div class="numeric-vector">['+Array.from(x[2].slice(0,4),v=>num(v,2)).join(', ')+', …]</div><p>'+x[3]+'</p></div>').join('');
  $('attention-map').setAttribute('aria-label','query帧 '+(state.frame+1)+'，key帧 '+(state.key+1)+'，注意力权重 '+num(row[state.key]*100,3)+'%');
}
function drawWeightRow(row) {
  const {c,w,h,canvas}=context('attention-row');const p={x:10,y:12,w:w-20,h:h-36,kind:'weights',cols:row.length},max=Math.max(...row)*1.12;canvas._plot=p;
  c.strokeStyle=colors.line;c.beginPath();c.moveTo(p.x,p.y+p.h);c.lineTo(p.x+p.w,p.y+p.h);c.stroke();
  for(let k=0;k<row.length;k++){const bar=row[k]/max*p.h;c.fillStyle=k===state.key?colors.orange:'#90af7a';c.fillRect(p.x+k/row.length*p.w,p.y+p.h-bar,Math.max(1,p.w/row.length-.5),bar);}
  c.fillStyle=colors.muted;c.font='8px '+mono;c.fillText('key 001',p.x,h-4);c.textAlign='right';c.fillText('186',w-10,h-4);c.textAlign='left';c.fillText('当前 key '+pad(state.key+1)+' · '+num(row[state.key]*100,2)+'%',p.x,9);
}
function getMask() {
  const key=[state.version,state.maskMode,state.probability,state.seed,state.error,state.alpha].join(':');
  if(!maskCache||maskCache.key!==key)maskCache={key,result:makeMaskedExample(data.stft.values,{mode:state.maskMode,probability:state.probability,seed:state.seed,error:state.error,alpha:state.alpha})};return maskCache.result;
}
function renderMask() {
  const m=getMask();paintHeat('mask-target',m.target);paintHeat('mask-input',m.input,{supervision:m.mask});paintHeat('mask-prediction',m.prediction);paintHeat('mask-error',m.errors,{error:true,mask:m.mask});
  $('mask-count-label').textContent='监督 '+num(m.count/(186*40)*100,1)+'% / '+m.count+' 点';
  $('loss-base').textContent=num(m.l1,4);$('loss-high').textContent=num((m.highL1??0)*state.alpha,4);$('loss-total').textContent=num(m.loss,4);
  const t=state.frame,f=state.band,selected=!!m.mask[t][f],high=!!m.high[t][f];
  $('mask-inspector').innerHTML='<span>检查 t'+pad(t+1)+' / '+num(data.stft.frequency[f],2)+' Hz</span><span>y = <b>'+num(m.target[t][f])+'</b></span><span>ŷ = <b>'+num(m.prediction[t][f])+'</b></span><span>|y−ŷ| = <b>'+num(m.errors[t][f])+'</b></span><span>监督 M: <b>'+(selected?'是':'否')+'</b> · 高幅值 C: <b>'+(high?'是':'否')+'</b></span><span>|M| = '+m.count+' · |C| = '+m.highCount+'</span>';
  if(m.highCount===0)$('mask-inspector').insertAdjacentHTML('beforeend','<span>高幅值子集为空，本演示额外项按 0 处理；源码空均值会得到 NaN。</span>');
}
function renderOutputs() {
  const mode=state.output;
  const stages=[['STFT 特征','[1, 186, 40]'],['输入投影 + PE','[1, 186, 768]'],['Transformer × 6','[1, 186, 768]']];
  if(mode==='reconstruction')stages.push(['重建预测头','[1, 186, 40]']);
  if(mode==='classification'){stages.push(['中心 10 帧均值','[1, 768]']);stages.push(['分类头','[1, 1]']);}
  $('output-pipeline').innerHTML=stages.map((x,i)=>'<div class="flow-block '+(i===stages.length-1?'active':'')+'"><strong>'+x[0]+'</strong><span>'+x[1]+'</span></div>').join('');
  const details={
    embedding:{shape:'[1, 186, 768]',title:'每帧一个 768 维上下文表示',result:'保留时间序列，供后续任务使用；没有自动汇总整段信号。',code:'h = self.transformer(x,\n    src_key_padding_mask=pad_mask)\nh = h.transpose(0, 1)\n\nif intermediate_rep:\n    return h  # [B, T, 768]',note:'色块只示意输出结构（绘制前 48 维），不是预训练 Transformer 输出。',explanation:'intermediate_rep=True 时在这里返回。768 维上下文表示由 Transformer 产生；.pth 文件存的是生成它的参数。'},
    reconstruction:{shape:'[1, 186, 40]',title:'还原每一帧的 40 维频率特征',result:'预测值与原始时频目标比较，仅在选中的监督位置计算重建 loss。',code:'h = self.transformer(x,\n    src_key_padding_mask=pad_mask)\nh = h.transpose(0, 1)\n\npred = self.spec_prediction_head(h)\nreturn pred, pos_enc',note:'此处展示目标 + 可控误差生成的模拟重建，不是预训练模型预测。',explanation:'预测头包含 Linear(768,768) → GELU → LayerNorm → Linear(768,40)。它将表示映射回时频特征空间。'},
    classification:{shape:'[1, 1]',title:'把时间序列汇总成一个分类输入',result:'新增 Linear(768,1) 输出 logit；训练用 BCEWithLogitsLoss，预测概率时再 sigmoid。',code:'h = upstream(inputs, pad_mask,\n    intermediate_rep=True)\nmiddle = h.shape[1] // 2\nz = h[:, middle-5:middle+5].mean(1)\nlogits = self.linear_out(z)\n# [B,T,768] → [B,768] → [B,1]',note:'下图示意取中间 10 帧的均值；分类概率与 BCE 在下一节检查。',explanation:'仓库采用中心 10 帧池化，而非所有时间帧求平均。分类训练不经过重建预测头；分类头需要带标签数据训练。'}
  };
  const d=details[mode];$('output-result').innerHTML='<span>'+d.result+'</span><strong>'+d.shape+'</strong>';$('output-detail-title').textContent=d.title;$('code-source').textContent=mode==='classification'?'finetune_model.py':'masked_tf_model.py';$('output-code').textContent=d.code;$('output-explanation').textContent=d.explanation;$('output-visual-note').textContent=d.note;
  if(mode==='reconstruction')paintHeat('output-visual',getMask().prediction);
  else if(mode==='embedding')paintHeat('output-visual',getEmbeddings().map(row=>Array.from(row.slice(0,48))),{dimensions:true});
  else drawPool();
}
function drawPool() {
  const {c,w,h}=context('output-visual');const x=20,y=35,pw=w-40,ph=80,T=186,mid=93;
  c.fillStyle=colors.muted;c.font='9px '+mono;c.fillText('时间序列：186 个 embedding',x,18);
  for(let t=0;t<T;t++){c.fillStyle=t>=mid-5&&t<mid+5?colors.green:'#dfe8d5';c.fillRect(x+t/T*pw,y,pw/T-0.5,ph);}
  c.fillStyle=colors.green;c.textAlign='center';c.fillText('取帧 '+(mid-5+1)+'–'+(mid+5)+'，沿时间求均值',w/2,y+ph+24);
  c.strokeStyle='#98b58b';c.beginPath();c.moveTo(x+(mid-5)/T*pw,y+ph+3);c.lineTo(w/2-32,y+ph+55);c.moveTo(x+(mid+5)/T*pw,y+ph+3);c.lineTo(w/2+32,y+ph+55);c.stroke();
  c.fillStyle='#e8f2de';c.fillRect(w/2-80,y+ph+56,160,35);c.fillStyle=colors.green;c.fillText('一个 768 维分类输入',w/2,y+ph+78);c.textAlign='left';
}
function render() {
  if(!data)return;syncFrame();
  if(state.page===1)renderSTFT();if(state.page===2)renderTokens();if(state.page===3)renderAttention();if(state.page===4)renderMask();if(state.page===5)renderOutputs();
}
function queueRender(){if(frameRAF)return;frameRAF=requestAnimationFrame(()=>{frameRAF=0;render();});}
function stopPlay(){if(playTimer){clearInterval(playTimer);playTimer=null;}$('play').textContent='▶';$('play').setAttribute('aria-label','播放时间帧');}
function goPage(index,scroll=true) {
  state.page=clamp(index,0,pages.length-1);stopPlay();const p=pages[state.page];
  document.querySelectorAll('[data-lesson]').forEach((el,i)=>el.hidden=i!==state.page);
  document.querySelectorAll('[data-page]').forEach((el,i)=>{el.classList.toggle('active',i===state.page);el.setAttribute('aria-current',i===state.page?'step':'false');});
  $('chapter-kicker').textContent=String(state.page+1).padStart(2,'0')+' / '+p[1];$('chapter-title').textContent=p[2];$('chapter-desc').textContent=p[3];
  $('chapter-count').innerHTML=String(state.page+1).padStart(2,'0')+'<span>/ 07</span>';$('lesson-progress').textContent=String(state.page+1).padStart(2,'0')+' / 07 · '+p[4];
  $('previous').disabled=state.page===0;$('next').textContent=state.page===6?'回到模型架构 ↺':'下一节：'+['模型架构','波形与时频图','从频谱到 token','多头注意力','遮挡与 loss','取表示还是预测','分类与微调'][state.page+1]+' →';
  $('sample-bar').hidden=state.page===0||state.page===6;$('synthetic-controls').hidden=state.page===0||state.page===6||state.source!=='synthetic';
  if(location.hash!=='#'+p[0])history.replaceState(null,'','#'+p[0]);
  if(scroll)window.scrollTo({top:0,behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
  requestAnimationFrame(render);
}
function setFrame(i){state.frame=clamp(Math.round(i),0,185);queueRender();}
function pointerPosition(event,canvas){const r=canvas.getBoundingClientRect();return{x:event.clientX-r.left,y:event.clientY-r.top};}
function bindCanvas(id,handler){const el=$(id);let pressed=false;const apply=e=>{if(!el._plot)return;const pos=pointerPosition(e,el);handler(pos,el._plot);};el.addEventListener('pointerdown',e=>{pressed=true;el.setPointerCapture(e.pointerId);apply(e);});el.addEventListener('pointermove',e=>{if(pressed)apply(e);});el.addEventListener('pointerup',()=>pressed=false);el.addEventListener('pointercancel',()=>pressed=false);}
function heatSelect(pos,p){state.frame=clamp(Math.floor((pos.x-p.x)/p.w*p.cols),0,p.cols-1);state.band=clamp(p.rows-1-Math.floor((pos.y-p.y)/p.h*p.rows),0,39);queueRender();}
function resetComputed(){state.version++;embCache.clear();attentionCache.clear();maskCache=null;}
function synthetic(){
  const f1=+$('freq-a').value,f2=+$('freq-b').value,amp=+$('amp-b').value;
  $('freq-a-value').textContent=f1+' Hz';$('freq-b-value').textContent=f2+' Hz';$('amp-b-value').textContent=num(amp,2);
  const wave=Array.from({length:10240},(_,i)=>{const t=i/2048;const low=1+.5*Math.sin(2*Math.PI*.35*t),high=.15+Math.exp(-Math.pow((t-2.6)/.8,2));const noise=.035*(Math.sin(i*1.173)+Math.cos(i*2.417));return low*Math.sin(2*Math.PI*f1*t)+amp*high*Math.sin(2*Math.PI*f2*t+.5)+noise;});
  data={sample:{...original.sample,waveform:wave,unit:'a.u.'},stft:stftWaveform(wave),meta:{containsModelInference:false}};
  resetComputed();$('wave-unit').textContent='（合成信号，任意单位）';render();
}
function wire(){
  document.querySelectorAll('[data-page]').forEach(b=>b.addEventListener('click',()=>goPage(+b.dataset.page)));
  $('previous').addEventListener('click',()=>goPage(state.page-1));$('next').addEventListener('click',()=>goPage((state.page+1)%pages.length));
  $('frame-slider').addEventListener('input',e=>setFrame(+e.target.value));
  $('stft-view').addEventListener('change',e=>{state.spectrumView=e.target.value;render();});
  $('play').addEventListener('click',()=>{if(playTimer){stopPlay();return;}$('play').textContent='Ⅱ';$('play').setAttribute('aria-label','暂停时间帧');playTimer=setInterval(()=>setFrame((state.frame+1)%186),110);});
  $('signal-source').addEventListener('change',e=>{stopPlay();state.source=e.target.value;$('synthetic-controls').hidden=state.source!=='synthetic';if(state.source==='synthetic')synthetic();else{data=original;resetComputed();$('wave-unit').textContent='（单位依据原 demo 标注：µV）';render();}});
  let synthTimer;for(const id of ['freq-a','freq-b','amp-b'])$(id).addEventListener('input',()=>{clearTimeout(synthTimer);synthTimer=setTimeout(synthetic,60);});
  $('info-button').addEventListener('click',()=>$('info-dialog').showModal());$('close-info').addEventListener('click',()=>$('info-dialog').close());$('info-dialog').addEventListener('click',e=>{if(e.target===$('info-dialog')){const r=e.target.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.target.close();}});
  bindCanvas('waveform',(pos,p)=>{const t=clamp((pos.x-p.x)/p.w,0,1)*5;setFrame((t-data.stft.time[0])/(50/2048));});
  for(const id of ['spectrogram','token-heatmap','mask-target','mask-input','mask-prediction','mask-error'])bindCanvas(id,heatSelect);
  for(const id of ['spectrum','input-vector'])bindCanvas(id,(pos,p)=>{state.band=clamp(Math.floor((pos.x-p.x)/p.w*40),0,39);queueRender();});
  $('token-strip').innerHTML=Array.from({length:186},(_,i)=>'<button data-token="'+i+'" title="时间帧 '+(i+1)+'" aria-label="选择时间帧 '+(i+1)+'"></button>').join('');
  $('token-strip').addEventListener('click',e=>{const b=e.target.closest('[data-token]');if(b)setFrame(+b.dataset.token);});
  $('projection-stages').addEventListener('click',e=>{const b=e.target.closest('[data-stage]');if(!b)return;state.projection=b.dataset.stage;$('projection-stages').querySelectorAll('button').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b));});render();});
  $('head-selector').innerHTML=Array.from({length:12},(_,i)=>'<button data-head="'+i+'" aria-pressed="'+(i===0)+'" class="'+(i===0?'selected':'')+'">head '+String(i+1).padStart(2,'0')+'</button>').join('');
  $('head-selector').addEventListener('click',e=>{const b=e.target.closest('[data-head]');if(!b)return;state.head=+b.dataset.head;$('head-selector').querySelectorAll('button').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b));});render();});
  $('temperature').addEventListener('input',e=>{state.temperature=+e.target.value;$('temperature-value').textContent=num(state.temperature,1);queueRender();});
  $('padding-toggle').addEventListener('change',e=>{state.padding=e.target.checked?20:0;render();});
  bindCanvas('attention-map',(pos,p)=>{state.frame=clamp(Math.floor((pos.y-p.y)/p.h*186),0,185);state.key=clamp(Math.floor((pos.x-p.x)/p.w*186),0,185);queueRender();});
  bindCanvas('attention-row',(pos,p)=>{state.key=clamp(Math.floor((pos.x-p.x)/p.w*186),0,185);queueRender();});
  $('attention-top').addEventListener('click',e=>{const b=e.target.closest('[data-key]');if(b){state.key=+b.dataset.key;render();}});
  $('mask-modes').addEventListener('click',e=>{const b=e.target.closest('[data-mask]');if(!b)return;state.maskMode=b.dataset.mask;$('mask-modes').querySelectorAll('button').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b));});render();});
  for(const [id,key,output] of [['mask-prob','probability','mask-prob-value'],['error-scale','error','error-scale-value'],['loss-alpha','alpha','loss-alpha-value']])$(id).addEventListener('input',e=>{state[key]=+e.target.value;$(output).textContent=num(state[key],2);queueRender();});
  $('remask').addEventListener('click',()=>{state.seed++;render();});
  $('output-modes').addEventListener('click',e=>{const b=e.target.closest('[data-output]');if(!b)return;state.output=b.dataset.output;$('output-modes').querySelectorAll('button').forEach(x=>{x.classList.toggle('selected',x===b);x.setAttribute('aria-pressed',String(x===b));});render();});
  window.addEventListener('resize',queueRender);window.addEventListener('hashchange',()=>{const i=pages.findIndex(x=>'#'+x[0]===location.hash);if(i>=0&&i!==state.page)goPage(i,false);});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopPlay();});
}

async function boot(){
  try{const response=await fetch('./brain-data.json');if(!response.ok)throw new Error('样本数据加载失败：'+response.status);original=await response.json();data=original;if(data.stft.values.length!==186||data.stft.values[0].length!==40)throw new Error('样本形状与预期不一致');
    mountFineTune($('finetune-container'));mountArchitecture($('architecture-container'));wire();const index=pages.findIndex(x=>'#'+x[0]===location.hash);goPage(index<0?0:index,false);$('loading').hidden=true;
  }catch(error){console.error(error);$('loading').innerHTML='<div class="error-message"><b>页面还没能加载完整。</b><p>'+escape(error.message)+'</p><button class="btn" onclick="location.reload()">重新加载</button></div>';}
}
boot();
