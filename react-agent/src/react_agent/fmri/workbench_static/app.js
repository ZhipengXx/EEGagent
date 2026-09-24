const state = {
  runs: [],
  selected: null,
  view: null,
  token: 0,
  tab: "overview",
  mode: "contrast",
  frame: 0,
  clock: "absolute",
  montage: false,
  fit: true,
  playing: false,
  timer: null,
  paintToken: 0,
  shownFrame: null,
  shownMode: null,
  shownRun: null,
  shownUrls: null,
  followJob: false,
  traceIndex: 0,
  configs: null,
  surface: "home",
  retrieval: {
    dataset: "eeg",
    exp_setting: "intra-subject",
    subject: "",
    epochs: "50",
    seed: "0",
    batch_size: "1024",
    lr: "",
    gpu_seconds: "28800",
    gpu: "",
    data_root: "",
    train_dir: "",
    test_dir: "",
    stop: "chain_early",
    policy: "adaptive",
    training_strategy: "pooled_subjects",
    generalization_target: "",
    held_out_subjects: "",
    gpu_mode: "auto_one",
  },
  discovered: false,
  shell: null,
  consoleLines: [],
  retrievalAction: "",
  busy: false,
  phase: "idle",
  progress: null,
  refused: false,
  trainStatus: null,
  trainPoll: null,
};

const main = document.getElementById("main");
const listEl = document.getElementById("run-list");

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function badge(label, tone) {
  return `<span class="badge ${esc(tone || "neutral")}">${esc(label || "未记录")}</span>`;
}

async function loadRuns() {
  const res = await fetch("/api/runs");
  const payload = await res.json();
  state.runs = payload.runs || [];
  renderList();
  if (state.surface === "home") {
    renderHome();
    return;
  }
  if (state.surface !== "screen") return;
  if (!state.selected && state.runs.length) selectRun(state.runs[0].run_id);
  if (!state.runs.length) renderEmpty();
}

function renderHome() {
  const recent = state.runs.slice(0, 5);
  const rows = recent.map((row) => `
    <button type="button" class="recent-row" data-run="${esc(row.run_id)}">
      <span class="recent-name">${esc(row.sample_id || row.run_id.split("/").pop())}</span>
      ${badge(row.decision_label, row.decision_tone)}
      <span class="recent-time">${esc(row.file_mtime)}</span>
    </button>`).join("");
  main.innerHTML = `
    <section class="home">
      <header class="home-head">
        <h1>要做什么？</h1>
        <p class="muted">数值检查只评估生成信号的数值一致性，不代表真实脑响应已验证。</p>
      </header>
      <div class="task-grid">
        <article class="task-card">
          <h2>查看检查结果</h2>
          <p class="muted">看每次检查是否通过，哪些地方需要关注。</p>
          <button type="button" class="primary" id="home-results"${state.runs.length ? "" : " disabled"}>查看最近结果</button>
        </article>
        <article class="task-card">
          <h2>新建数值检查</h2>
          <p class="muted">填写服务器上的图片路径，生成信号后自动做数值检查。</p>
          <button type="button" class="primary" id="home-new">新建检查</button>
        </article>
        <article class="task-card">
          <h2>EEG / MEG 检索研究</h2>
          <p class="muted">选好数据和被试，由 agent 规划并运行检索实验。</p>
          <button type="button" class="primary" id="home-research">进入检索实验</button>
        </article>
      </div>
      <section class="card">
        <h2>最近结果</h2>
        <div class="recent-list">${rows || `<p class="muted">还没有检查记录。先新建一次检查。</p>`}</div>
      </section>
    </section>`;
  const results = document.getElementById("home-results");
  if (results) results.addEventListener("click", () => {
    if (!state.runs.length) return;
    state.selected = state.runs[0].run_id;
    setSurface("screen");
  });
  document.getElementById("home-new").addEventListener("click", openDrawer);
  document.getElementById("home-research").addEventListener("click", () => setSurface("retrieval"));
  main.querySelectorAll(".recent-row").forEach((button) => {
    button.addEventListener("click", () => {
      state.selected = button.dataset.run;
      setSurface("screen");
    });
  });
}

function renderList() {
  const q = document.getElementById("search").value.trim().toLowerCase();
  const rows = state.runs.filter((row) => {
    const hay = `${row.sample_id || ""} ${row.run_id || ""}`.toLowerCase();
    return !q || hay.includes(q);
  });
  const groups = new Map();
  rows.forEach((row) => {
    const key = row.sample_id || "未命名";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  });
  listEl.innerHTML = [...groups.entries()].map(([name, items]) => `
    <section class="run-group">
      <h2>${esc(name)}</h2>
      ${items.map((row) => `
        <button type="button" class="run-item${row.run_id === state.selected ? " active" : ""}" data-run="${esc(row.run_id)}" title="${esc(row.decision_title || row.decision_label)}">
          <span class="run-name">${esc(row.run_id.split("/").pop())}</span>
          ${badge(row.decision_label, row.decision_tone)}
          <span class="run-meta">${esc(row.file_mtime)} ${esc(row.file_time_note || "文件修改时间")} · ${esc(row.mode_label)}</span>
        </button>`).join("")}
    </section>`).join("") || `<p class="muted">没有匹配的记录</p>`;
  listEl.querySelectorAll(".run-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.followJob = false;
      document.getElementById("sidebar").classList.remove("open");
      selectRun(button.dataset.run);
    });
  });
}

function renderEmpty() {
  main.innerHTML = `<section class="card"><h1>还没有检查记录</h1><p class="muted">开始一次检查后，结果会出现在这里。</p><button type="button" id="empty-new" class="primary">新建检查</button></section>`;
  document.getElementById("empty-new").addEventListener("click", openDrawer);
}

async function selectRun(runId) {
  state.selected = runId;
  state.token += 1;
  const token = state.token;
  state.playing = false;
  clearTimeout(state.timer);
  state.paintToken += 1;
  state.shownFrame = null;
  state.shownMode = null;
  state.shownRun = null;
  state.shownUrls = null;
  renderList();
  main.innerHTML = `<p class="muted">正在加载记录</p>`;
  const res = await fetch("/api/view?run=" + encodeURIComponent(runId));
  if (token !== state.token) return;
  if (!res.ok) {
    main.innerHTML = `<p class="error">记录无法读取</p>`;
    return;
  }
  state.view = await res.json();
  if (token !== state.token || state.surface !== "screen") return;
  state.frame = 0;
  state.mode = (state.view.brain.modes || []).includes("contrast") ? "contrast" : (state.view.brain.modes[0] || "raw");
  state.montage = !(state.view.brain.modes || []).length;
  state.traceIndex = 0;
  document.getElementById("header-status").innerHTML = badge(state.view.program.label, "neutral");
  renderMain();
}

function renderMain() {
  const view = state.view;
  if (!view || view.run_id !== state.selected) return;
  const tabs = [
    ["overview", "结论"],
    ["brain", "脑图"],
    ["metrics", "指标"],
    ["trace", "技术详情"],
  ];
  if (state.tab === "files") state.tab = "trace";
  main.innerHTML = `
    <div class="page">
      <div class="tabs">${tabs.map(([id, label]) => `<button type="button" class="${state.tab === id ? "active" : ""}" data-tab="${id}">${label}</button>`).join("")}</div>
      <div id="tab-body"></div>
    </div>`;
  main.querySelectorAll("[data-tab]").forEach((button) => {
    button.addEventListener("click", () => { state.tab = button.dataset.tab; renderMain(); });
  });
  const body = document.getElementById("tab-body");
  if (state.tab === "overview") body.innerHTML = overviewHtml(view);
  if (state.tab === "brain") body.innerHTML = brainHtml(view);
  if (state.tab === "metrics") body.innerHTML = metricsHtml(view);
  if (state.tab === "trace") body.innerHTML = techHtml(view);
  bindTab(view);
}

function attentionItems(view) {
  const rows = (view.coverage || []).filter((row) => row.blocks_completion || row.tone === "warning");
  const items = rows.map((row) => `<li><span>${esc(row.label_zh || row.id)}</span>${badge(row.status_label, row.tone)}</li>`);
  if (view.followup_note) items.push(`<li><span>${esc(view.followup_note)}</span></li>`);
  return items.length ? `<ul class="attention-list">${items.join("")}</ul>` : `<p class="muted">没有需要关注的事项。</p>`;
}

function overviewHtml(view) {
  const sample = view.sample;
  const done = view.coverage_satisfied;
  const total = view.coverage_required;
  const width = total ? Math.round((done / total) * 100) : 0;
  const tone = view.decision.verdict_tone || "neutral";
  return `
    <section class="verdict-banner ${esc(tone)}">
      <div class="verdict-main">
        <span class="verdict-word">${esc(view.decision.short_label || view.decision.verdict_label)}</span>
        <div>
          <h1>${esc(view.sample_id)}</h1>
          <p>${esc(view.decision.verdict_label)}${view.decision.stop_label ? " · " + esc(view.decision.stop_label) : ""}</p>
          <p class="depth-line">${view.screen_depth === "deep" ? "深层筛查" : "标准筛查"} · 实际到达 ${esc(view.depth_reached || "层级未记录")}</p>
        </div>
      </div>
      <p class="scope">仅评估生成信号的数值一致性。</p>
      <p class="muted">通过不代表真实脑响应已验证。</p>
    </section>
    <section class="two">
      <article class="card">
        <h2>需关注事项</h2>
        ${attentionItems(view)}
      </article>
      <article class="card">
        <h2>必需检查</h2>
        <p class="check-count"><b>${done} / ${total}</b> 已完成</p>
        <div class="progress-track"><div class="progress-fill" style="width:${width}%"></div></div>
        <p class="muted">${done === total ? "必需检查已完成。" : "必需检查未完成。"}</p>
      </article>
    </section>
    <section class="card">
      ${brainPreview(view)}${stimulusThumb(view)}
      <p class="muted">${esc(shapeText(sample.shape))} · ${esc(view.headline || sample.protocol_text)}</p>
    </section>
    <section class="card">
      <details>
        <summary>全部检查项目</summary>
        ${coverageTable(view)}
      </details>
    </section>`;
}

function techHtml(view) {
  const sample = view.sample;
  const llm = view.llm;
  const cost = llm.api_usd == null ? "费用未提供" : llm.api_usd;
  const depth = view.depth_reached || "层级未记录";
  return `
    <section class="grid">
      <article class="card stat"><span class="muted">检查深度</span><b>${esc(depth)}</b><span class="muted">${view.tool_count} 次工具</span></article>
      <article class="card stat"><span class="muted">调用成本</span><b>${llm.lm_calls == null ? "未提供" : llm.lm_calls}</b><span class="muted">LLM 次数 · ${esc(cost)}</span></article>
      <article class="card stat"><span class="muted">模式</span><b>${esc(view.mode_label)}</b><span class="muted">analysis_goal ${esc(view.analysis_goal || "未记录")}</span></article>
      <article class="card stat"><span class="muted">刺激起点</span><b>${sample.onset_s == null ? "未记录" : "t_video − " + sample.onset_s + "s"}</b><span class="muted">${esc(sample.protocol_text)}</span></article>
    </section>
    ${view.screen_depth === "deep" ? `<p class="muted">参考分布只有在配置开启参考打分、且样本库数量足够时才会影响结论；否则只描述离群程度。</p>` : ""}
    <h2 class="section-title">Agent 轨迹</h2>
    ${traceHtml(view)}
    <h2 class="section-title">产物与记忆</h2>
    ${filesHtml(view)}`;
}

function shapeText(shape) {
  if (!Array.isArray(shape) || shape.length < 2) return "形状未记录";
  return `${shape[0]} 个时间点 × ${Number(shape[1]).toLocaleString("en-US")} 个顶点`;
}

function coverageTable(view) {
  if (!view.coverage.length) return `<p class="muted">没有问题覆盖记录</p>`;
  return `<table class="coverage"><tbody>${view.coverage.map((row) => `<tr><td><div>${esc(row.label_zh || row.id)}</div><details><summary>内部 id</summary><span class="path">${esc(row.id)}</span></details></td><td>${esc(row.required_label)}</td><td class="status-cell">${badge(row.status_label, row.tone)}${row.blocks_completion ? " 阻塞完成" : ""}</td></tr>`).join("")}</tbody></table>`;
}

function stimulusThumb(view) {
  if (!view.sample.stimulus_rel) return "";
  const src = `/runs/${encodeURI(view.run_id)}/${view.sample.stimulus_rel.split("/").map(encodeURIComponent).join("/")}`;
  return `<p class="muted">刺激图</p><img class="brain-img" alt="刺激图缩略图" src="${src}">`;
}

function brainPreview(view) {
  if (!(view.brain.modes || []).length) {
    return view.brain.montage
      ? `<h2>脑图拼图</h2><p class="muted">当前记录只有拼图，没有逐帧数组。</p><img class="brain-img" alt="全部帧拼图" src="/runs/${encodeURI(view.run_id)}/brain_tstrip.png">`
      : `<p class="muted">没有脑图数组。</p>`;
  }
  return `<h2>当前帧</h2><p class="muted" id="frame-caption"></p><p class="muted" id="frame-status"></p>${stageBody(view)}`;
}

function stageBody(view) {
  const picture = state.montage
    ? `<img id="montage-img" class="brain-img" alt="全部帧拼图" src="/runs/${encodeURI(view.run_id)}/brain_tstrip.png">`
    : viewGrid();
  return `<div class="stage-row">${picture}${scaleHtml(view)}</div>`;
}

function viewGrid() {
  return `<div class="views">${["left", "right", "posterior"].map((name) => `<img class="view-img" data-view="${name}" alt="${name}">`).join("")}</div>`;
}

function brainHtml(view) {
  const brain = view.brain;
  if (!(brain.modes || []).length) return `<section class="card">${brainPreview(view)}</section>`;
  return `
    <section class="card">
      <h2>脑图</h2>
      <div class="toolbar">
        <div class="tool-group"><label>信号 <select id="signal-mode">${brain.modes.map((mode) => `<option value="${esc(mode)}"${mode === state.mode ? " selected" : ""}>${esc(mode)}</option>`).join("")}</select></label></div>
        <div class="tool-group button-group" role="group" aria-label="帧播放">
          <button type="button" id="prev-frame" aria-label="上一帧">上一帧</button>
          <button type="button" id="play-frame" aria-label="播放">播放</button>
          <button type="button" id="next-frame" aria-label="下一帧">下一帧</button>
        </div>
        <div class="tool-group">
          <div class="button-group" role="group" aria-label="显示大小">
            <button type="button" id="fit-frame">适应</button>
            <button type="button" id="full-frame">全屏</button>
          </div>
          <button type="button" id="montage-toggle" class="toggle" aria-pressed="${state.montage ? "true" : "false"}">${state.montage ? "单帧" : "全部帧"}</button>
        </div>
      </div>
      <p id="frame-caption" class="muted"></p>
      <p id="frame-status" class="muted"></p>
      <div id="brain-stage" class="stage${state.fit ? " fit" : ""}">
        ${stageBody(view)}
      </div>
      ${timelineHtml(view)}
      <div class="controls"><input id="frame-range" type="range" min="0" max="${brain.n_frames - 1}" value="${state.frame}" aria-label="帧"></div>
    </section>
    <section class="card"><h2>时序</h2>${seriesHtml(view)}</section>
    <section class="card"><h2>空间</h2>${roiHtml(view)}</section>`;
}

function scaleHtml(view) {
  const limits = (view.brain.color_limits || {})[state.mode];
  const vmin = limits ? _sigClient(limits.vmin) : "未提供";
  const vmax = limits ? _sigClient(limits.vmax) : "未提供";
  return `<div class="scale"><div class="scale-meter"><div class="scale-labels"><span>${esc(vmax)}</span><span>0</span><span>${esc(vmin)}</span></div><span class="gradient" aria-hidden="true"></span></div><p class="muted">${esc(view.brain.units)}</p><p class="muted">红/蓝表示正负，不代表脑响应已验证</p></div>`;
}

function timelineHtml(view) {
  const windows = view.windows || [];
  if (!windows.length) return "";
  const end = Math.max(...windows.map((item) => item.end_s));
  const segs = windows.map((item) => {
    const width = ((item.end_s - item.start_s) / end) * 100;
    const kind = item.label === "图像" ? "image" : "gray";
    return `<span class="win ${kind}" style="width:${width}%">${esc(item.label)}</span>`;
  }).join("");
  const marks = view.sample.protocol_kind === "image16"
    ? `<span class="mark" style="left:${(4 / end) * 100}%">4s</span><span class="mark" style="left:${(5 / end) * 100}%">5s</span>`
    : "";
  return `<div class="timeline">${segs}</div><div class="marks">${marks}</div>`;
}

function _sigClient(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "未提供";
  return num.toPrecision(3);
}

function seriesHtml(view) {
  const series = view.series;
  if (!series.available) return `<p class="muted">${esc(series.reason)}</p>`;
  return `<div class="controls"><label>曲线 <select id="series-kind"><option value="R">R contrast RMS</option><option value="A">A 相邻 RMS 变化</option><option value="S">S 顶点模式变化</option></select></label></div><svg id="chart" class="chart"></svg><p class="muted">${esc(series.note)} 点击点会换到对应帧。直线连接，没有平滑。</p><div id="series-table"></div>`;
}

function roiHtml(view) {
  const roi = view.roi;
  if (!roi.available) return `<p class="muted">${esc(roi.reason)}</p>`;
  return `<p class="muted">${esc(roi.note)}</p><table><thead><tr><th>ROI</th><th>名称</th><th>signed mean</th><th>RMS</th></tr></thead><tbody>${roi.rows.map((row) => `<tr><td>${esc(row.roi_key)}</td><td>${esc(row.name)}</td><td>${fmt(row.signed_mean)}</td><td>${fmt(row.rms)}</td></tr>`).join("")}</tbody></table>`;
}

function metricsHtml(view) {
  const preferred = ["输入与格式", "分布统计", "灰屏对照", "时序特征", "空间特征", "模型和参考评估", "运行成本", "其他"];
  const groups = new Map();
  view.metrics.forEach((row, index) => {
    const name = row.group || "其他";
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push({ row, index });
  });
  const order = [...groups.keys()].sort((a, b) => {
    const ia = preferred.indexOf(a);
    const ib = preferred.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  const body = order.map((name) => `
    <h3 class="group-title">${esc(name)}</h3>
    <table><tbody>${groups.get(name).map(({ row, index }) => `
      <tr class="metric" data-metric="${index}">
        <td>${esc(row.label)}</td>
        <td class="num">${esc(row.display)}</td>
        <td><button type="button" class="info" data-metric="${index}" aria-label="打开${esc(row.label)}的解释">i</button></td>
      </tr>`).join("")}</tbody></table>`).join("");
  return `<section class="card"><h2>指标</h2><p class="scope">仅评估生成信号的数值一致性。</p><p class="muted">通过不代表真实脑响应已验证。</p>${body}</section>`;
}

function traceHtml(view) {
  const steps = view.trace || [];
  return `<section class="card trace"><div class="trace-list">${steps.map((step, index) => `<button type="button" data-step="${index}" class="${index === state.traceIndex ? "active" : ""}">${esc(step.label)} · ${esc(step.tool_name || step.type)}</button>`).join("") || `<p class="muted">没有事件</p>`}</div><div id="step-detail"></div></section>`;
}

function filesHtml(view) {
  const groups = {};
  view.artifacts.forEach((item) => { (groups[item.group] ||= []).push(item); });
  const files = Object.entries(groups).map(([group, items]) => `<h2>${esc(group)}</h2><ul>${items.map((item) => `<li><a href="/api/artifact?run=${encodeURIComponent(view.run_id)}&file=${encodeURIComponent(item.file)}">${esc(item.file)}</a></li>`).join("")}</ul>`).join("");
  const memory = view.memory;
  return `<section class="card">${files || `<p class="muted">没有产物</p>`}<h2>记忆</h2><p>命中 ${memory.hit_count == null ? "未记录" : memory.hit_count}</p><p class="muted">${esc(memory.note)}</p><p class="path">${esc(view.run_id)} <button type="button" id="copy-run" class="btn-small">复制 run 路径</button></p></section>`;
}

function bindTab(view) {
  if (state.tab === "overview" || state.tab === "brain") bindBrain(view);
  if (state.tab === "brain") bindSeries(view);
  if (state.tab === "metrics") {
    main.querySelectorAll("tr.metric").forEach((row) => {
      row.addEventListener("click", () => openExplain(view.metrics[Number(row.dataset.metric)]));
    });
  }
  if (state.tab === "trace") {
    showStep(view);
    main.querySelectorAll("[data-step]").forEach((button) => {
      button.addEventListener("click", () => {
        state.traceIndex = Number(button.dataset.step);
        renderMain();
      });
    });
  }
  const copy = document.getElementById("copy-run");
  if (copy) copy.addEventListener("click", () => navigator.clipboard.writeText(view.run_id));
}

function frameCaption(view, index) {
  const n = view.brain.n_frames || 1;
  const point = (view.series.times || [])[index] || { index };
  const abs = point.absolute_time_s == null ? "时间未记录" : point.absolute_time_s + "s";
  const rel = point.stimulus_relative_time_s == null ? "未记录" : point.stimulus_relative_time_s + "s";
  return `第 ${index + 1}/${n} 帧（索引 ${index}）· 视频时间 ${abs} · 刺激相对时间 ${rel} · ${state.mode}`;
}

function frameUrl(view, name, index, mode) {
  return `/api/frame?run=${encodeURIComponent(view.run_id)}&mode=${encodeURIComponent(mode)}&index=${index}&view=${encodeURIComponent(name)}`;
}

function applyShown(view) {
  const urls = state.shownUrls;
  if (!urls) return;
  document.querySelectorAll("[data-view]").forEach((img) => {
    const name = img.dataset.view;
    if (!urls[name]) return;
    img.alt = `${name} ${state.shownMode} 帧 ${state.shownFrame}`;
    img.src = urls[name];
  });
  const caption = document.getElementById("frame-caption");
  if (caption && state.shownFrame != null) caption.textContent = frameCaption(view, state.shownFrame);
}

function loadTriplet(view, index, mode) {
  const names = ["left", "right", "posterior"];
  return Promise.all(names.map((name) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve([name, img.src]);
    img.onerror = () => reject(new Error(name));
    img.src = frameUrl(view, name, index, mode);
  }))).then((pairs) => Object.fromEntries(pairs));
}

function prefetchFrame(view, index) {
  const n = view.brain.n_frames || 0;
  if (index < 0 || index >= n) return;
  ["left", "right", "posterior"].forEach((name) => {
    const img = new Image();
    img.src = frameUrl(view, name, index, state.mode);
  });
}

function paintFrame(view) {
  const brain = view.brain;
  const n = brain.n_frames || 1;
  state.frame = Math.max(0, Math.min(n - 1, state.frame));
  const stage = document.getElementById("brain-stage");
  if (stage) stage.classList.toggle("fit", state.fit);
  const caption = document.getElementById("frame-caption");
  const status = document.getElementById("frame-status");
  if (state.montage) {
    if (caption) caption.textContent = "全部帧拼图。这不是滑块上的单帧。";
    if (status) status.textContent = "";
    return Promise.resolve();
  }
  const index = state.frame;
  const mode = state.mode;
  const same = state.shownRun === view.run_id && state.shownMode === mode && state.shownFrame === index && state.shownUrls;
  if (state.shownRun === view.run_id && state.shownMode === mode && state.shownUrls) applyShown(view);
  if (same) {
    if (status) status.textContent = "";
    const range = document.getElementById("frame-range");
    if (range) range.value = String(index);
    prefetchFrame(view, index + 1);
    return Promise.resolve();
  }
  const token = ++state.paintToken;
  if (status) status.textContent = `正在载入第 ${index + 1} 帧`;
  return loadTriplet(view, index, mode).then((urls) => {
    if (token !== state.paintToken) return;
    state.shownRun = view.run_id;
    state.shownMode = mode;
    state.shownFrame = index;
    state.shownUrls = urls;
    applyShown(view);
    if (status) status.textContent = "";
    const range = document.getElementById("frame-range");
    if (range) range.value = String(index);
    drawSeries(view);
    prefetchFrame(view, index + 1);
  }).catch(() => {
    if (token !== state.paintToken) return;
    if (status) status.textContent = "这一帧没有载入";
  });
}

function bindBrain(view) {
  const brain = view.brain;
  if (!(brain.modes || []).length) return;
  paintFrame(view);
  document.getElementById("signal-mode")?.addEventListener("change", (event) => {
    state.mode = event.target.value;
    state.montage = false;
    renderMain();
  });
  document.getElementById("prev-frame")?.addEventListener("click", () => { state.frame -= 1; paintFrame(view); });
  document.getElementById("next-frame")?.addEventListener("click", () => { state.frame += 1; paintFrame(view); });
  document.getElementById("frame-range")?.addEventListener("input", (event) => {
    state.frame = Number(event.target.value);
    paintFrame(view);
  });
  document.getElementById("play-frame")?.addEventListener("click", () => togglePlay(view));
  document.getElementById("fit-frame")?.addEventListener("click", () => { state.fit = !state.fit; paintFrame(view); });
  document.getElementById("full-frame")?.addEventListener("click", () => document.getElementById("brain-stage")?.requestFullscreen?.());
  document.getElementById("montage-toggle")?.addEventListener("click", () => {
    if (!view.brain.montage && !state.montage) return;
    state.montage = !state.montage;
    renderMain();
  });
}

function togglePlay(view) {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) return;
  state.playing = !state.playing;
  const button = document.getElementById("play-frame");
  if (button) button.textContent = state.playing ? "暂停" : "播放";
  clearTimeout(state.timer);
  if (!state.playing) return;
  const loop = async () => {
    if (!state.playing) return;
    const n = view.brain.n_frames || 1;
    const ready = state.shownRun === view.run_id && state.shownMode === state.mode && state.shownFrame === state.frame;
    if (ready) state.frame = (state.frame + 1) % n;
    await paintFrame(view);
    if (!state.playing) return;
    state.timer = setTimeout(loop, 800);
  };
  loop();
}

function bindSeries(view) {
  drawSeries(view);
  document.getElementById("series-kind")?.addEventListener("change", () => drawSeries(view));
}

function drawSeries(view) {
  const svg = document.getElementById("chart");
  const series = view.series;
  if (!svg || !series.available) return;
  const kind = document.getElementById("series-kind")?.value || "R";
  const values = series[kind] || [];
  if (!values.length) {
    svg.innerHTML = "";
    return;
  }
  const width = 640;
  const height = 180;
  const pad = 28;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const coords = values.map((value, index) => {
    const x = pad + (index / Math.max(values.length - 1, 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return [x, y, index];
  });
  const d = coords.map((point, index) => `${index ? "L" : "M"}${point[0].toFixed(1)},${point[1].toFixed(1)}`).join(" ");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const active = state.shownFrame;
  svg.innerHTML = `<path d="${d}" fill="none" stroke="${cssVar("--accent", "#0e7490")}" stroke-width="2"></path>` + coords.map((point) => `<circle cx="${point[0]}" cy="${point[1]}" r="${point[2] === active ? 6 : 4}" data-index="${point[2]}"></circle>`).join("");
  svg.querySelectorAll("circle").forEach((dot) => {
    dot.addEventListener("click", () => {
      state.frame = Number(dot.dataset.index);
      state.montage = false;
      if (!document.querySelector("[data-view]")) {
        renderMain();
        return;
      }
      paintFrame(view);
    });
  });
  const table = document.getElementById("series-table");
  if (table) {
    table.innerHTML = `<details><summary>逐帧数值</summary><table><tbody>${values.map((value, index) => `<tr data-frame="${index}"><td>${index + 1}</td><td>${fmt(value)}</td></tr>`).join("")}</tbody></table></details>`;
  }
}

function showStep(view) {
  const step = (view.trace || [])[state.traceIndex];
  const box = document.getElementById("step-detail");
  if (!box || !step) return;
  box.innerHTML = `<h2>${esc(step.label)}</h2><p>${esc(step.tool_name || "")} · ${esc(step.status || "")}</p><p>${step.reason_missing ? "未记录选择理由" : esc(step.reason)}</p><p class="muted">类型 ${esc(step.type)} · 结果 ${esc(step.result_id || "未记录")}</p><p class="muted">${step.parent_call_id ? "parent " + esc(step.parent_call_id) : "旧事件没有层级，这里是按时间排列的平面记录。"}</p>`;
}

function openExplain(metric) {
  const def = metric.definition || {};
  document.getElementById("explain-body").innerHTML = `
    <p><b>${esc(metric.label)}</b> ${esc(metric.display)}</p>
    <h2>这是什么</h2><p>${esc(def.meaning || "")}</p>
    <h2>怎么计算</h2><p class="formula">${formatFormula(def.formula || "")}</p>
    <h2>可以说明什么</h2><p>${esc(def.meaning || "")}</p>
    <h2>不能说明什么</h2><ul>${(def.limitations || []).map((item) => `<li>${esc(item)}</li>`).join("")}</ul>
    <h2>阈值</h2><p>${def.threshold_kind === "none" ? "没有阈值" : esc(def.threshold_kind)}</p>
    <h2>来源</h2><p class="muted">execution ${esc(metric.execution_id || "未记录")} · signal ${esc(metric.signal_mode || "未记录")} · 字段 ${esc(metric.id)}</p>
    <pre class="path">${esc(JSON.stringify(metric.value, null, 2))}</pre>`;
  const panel = document.getElementById("explain");
  panel.hidden = false;
  document.getElementById("close-explain").focus();
}

function fmt(value) {
  if (value == null || Number.isNaN(value)) return "未提供";
  if (typeof value === "number") return value.toPrecision(4);
  return String(value);
}

function openDrawer() {
  document.getElementById("drawer").hidden = false;
  document.getElementById("image_path").focus();
  refreshModeNote();
}

function closeDrawer() {
  document.getElementById("drawer").hidden = true;
  document.getElementById("open-drawer").focus();
}

async function refreshModeNote() {
  if (!state.configs) {
    const res = await fetch("/api/configs");
    state.configs = await res.json();
    const select = document.getElementById("config");
    select.innerHTML = `<option value="">跟检查模式</option>` + (state.configs.modes || []).map((mode) => `<option value="${esc(mode.config)}">${esc(mode.label)}</option>`).join("");
  }
  const mode = document.getElementById("check_mode").value;
  const depth = document.getElementById("screen_depth");
  depth.disabled = mode !== "diagnostic";
  if (depth.disabled) depth.value = "standard";
  document.getElementById("depth-note").textContent = depth.disabled
    ? "深层筛查需要分层数值检查。快速数值检查不调用模型。"
    : depth.value === "deep"
      ? "更慢，调用接口次数更多。缺资源的项会标为检查被阻塞，不会算作通过。"
      : "只跑第一层检查。";
  const item = (state.configs.modes || []).find((row) => row.id === mode);
  document.getElementById("mode-note").textContent = item ? item.note : "";
  document.getElementById("mode-resolved").textContent = item ? item.resolved : "";
  const image = document.getElementById("image_path").value.trim();
  document.getElementById("image-preview").textContent = image ? image.split("/").pop() : "填写服务器上的图片路径。浏览器不能从本地选择框得到服务器路径。";
}

document.getElementById("toggle-side").addEventListener("click", () => {
  document.getElementById("sidebar").classList.toggle("open");
});
document.getElementById("open-drawer").addEventListener("click", openDrawer);
document.getElementById("close-drawer").addEventListener("click", closeDrawer);
document.getElementById("close-explain").addEventListener("click", () => { document.getElementById("explain").hidden = true; });
document.getElementById("search").addEventListener("input", renderList);
document.getElementById("check_mode").addEventListener("change", refreshModeNote);
document.getElementById("screen_depth").addEventListener("change", refreshModeNote);
document.getElementById("image_path").addEventListener("input", refreshModeNote);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    document.getElementById("drawer").hidden = true;
    document.getElementById("explain").hidden = true;
  }
});
document.getElementById("run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const error = document.getElementById("form-error");
  error.textContent = "";
  const body = new URLSearchParams(new FormData(event.target));
  if (!body.get("config")) body.delete("config");
  const res = await fetch("/run", { method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body });
  const job = await res.json();
  if (!res.ok) {
    error.textContent = job.error || job.message || "提交失败";
    return;
  }
  state.followJob = true;
  document.getElementById("header-status").innerHTML = badge("运行中", "info");
  pollJob();
});

async function pollJob() {
  const res = await fetch("/status");
  const job = await res.json();
  if (job.status === "running") {
    document.getElementById("header-status").innerHTML = badge("运行中", "info");
    setTimeout(pollJob, 2000);
    return;
  }
  document.getElementById("header-status").innerHTML = badge(job.status === "error" ? "任务失败" : "运行完成", job.status === "error" ? "danger" : "neutral");
  if (job.error) document.getElementById("form-error").textContent = job.error;
  if (job.status === "done" && job.view && state.followJob) {
    closeDrawer();
    state.selected = job.view;
    setSurface("screen");
    await loadRuns();
  }
}

function setSurface(name) {
  state.surface = name;
  document.body.dataset.surface = name;
  ["home", "screen", "retrieval"].forEach((id) => {
    document.getElementById("surface-" + id).classList.toggle("active", name === id);
  });
  document.getElementById("open-drawer").hidden = name === "retrieval";
  document.getElementById("toggle-side").hidden = name !== "screen";
  if (name === "retrieval") renderRetrieval();
  else if (name === "home") loadRuns();
  else if (state.selected) selectRun(state.selected);
  else loadRuns();
}

let agenticRequestId = "";

function retrievalBody(action) {
  const body = new URLSearchParams(state.retrieval);
  body.set("action", action);
  if (action === "run" && state.retrieval.policy === "agentic") {
    if (!agenticRequestId) agenticRequestId = `ui-${Date.now()}`;
    body.set("request_id", agenticRequestId);
  }
  return body;
}

function clearDiscovery() {
  state.discovered = false;
  state.subjects = [];
  state.gpus = [];
  state.retrieval.subject = "";
  state.retrieval.gpu = "";
}

const AGENTIC_STATUS = {
  created: "已创建", planning: "规划中", implementing: "编写代码", checking: "检查中", training: "训练中",
  analyzing: "分析结果", paused: "已暂停", finished: "已结束", blocked: "受阻", cancelled: "已停止",
};

const AGENTIC_DETAIL = {
  planner_decision_recorded_not_executed: "已记下决定，尚未执行",
};

function firstClause(text) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  if (!value) return "";
  const sentence = value.split(/[。！？\n]/)[0];
  return sentence.length > 72 ? `${sentence.slice(0, 72)}…` : sentence;
}

function hypothesisLine(hypothesis) {
  if (!hypothesis) return "尚未提出";
  return firstClause(hypothesis.statement || hypothesis.mechanism || hypothesis.observation) || "尚未提出";
}

function splitTopSlash(text) {
  let depth = 0;
  let at = -1;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "(") depth += 1;
    else if (ch === ")") depth -= 1;
    else if (ch === "/" && depth === 0) {
      if (at >= 0) return null;
      at = i;
    }
  }
  return at < 0 ? null : [text.slice(0, at), text.slice(at + 1)];
}

function replaceCall(text, name, render) {
  const token = `${name}(`;
  let out = "";
  let i = 0;
  while (i < text.length) {
    const at = text.indexOf(token, i);
    if (at < 0) {
      out += text.slice(i);
      break;
    }
    out += text.slice(i, at);
    let depth = 0;
    let j = at + name.length;
    for (; j < text.length; j += 1) {
      if (text[j] === "(") depth += 1;
      else if (text[j] === ")") {
        depth -= 1;
        if (depth === 0) {
          j += 1;
          break;
        }
      }
    }
    out += render(text.slice(at + token.length, j - 1));
    i = j;
  }
  return out;
}

function frac(top, bottom) {
  return `<span class="frac"><span>${top.trim()}</span><span>${bottom.trim()}</span></span>`;
}

function formatFormulaInner(text) {
  if (!text.includes("<")) {
    const parts = splitTopSlash(text);
    if (parts) return frac(formatFormulaInner(parts[0]), formatFormulaInner(parts[1]));
  }
  let out = replaceCall(text, "sqrt", (inner) => `√(${formatFormulaInner(inner)})`);
  out = replaceCall(out, "mean", (inner) => `mean(${formatFormulaInner(inner)})`);
  out = out.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, (_, top, bottom) => frac(formatFormulaInner(top), formatFormulaInner(bottom)));
  out = out.replace(/\\sqrt\{([^{}]*)\}/g, (_, inner) => `√(${formatFormulaInner(inner)})`);
  out = out.replace(/\\([a-zA-Z]+)/g, '<span class="math-op">$1</span>');
  out = out.replace(/\^\{([^{}]*)\}/g, "<sup>$1</sup>");
  out = out.replace(/\^([0-9A-Za-z]+)/g, "<sup>$1</sup>");
  out = out.replace(/_\{([^{}]*)\}/g, "<sub>$1</sub>");
  out = out.replace(/_([0-9]+|[A-Za-z]{1,2})(?![A-Za-z0-9])/g, "<sub>$1</sub>");
  return out;
}

function formatFormula(raw) {
  return `<span class="math">${formatFormulaInner(esc(raw || ""))}</span>`;
}

function renderProse(raw) {
  let text = esc(raw || "");
  text = text.replace(/\$([^$]+)\$/g, (_, inner) => `<span class="math">${formatFormulaInner(inner)}</span>`);
  text = text.replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, (_, top, bottom) => `<span class="math">${frac(formatFormulaInner(top), formatFormulaInner(bottom))}</span>`);
  text = text.replace(/\\sqrt\{([^{}]*)\}/g, (_, inner) => `<span class="math">√(${formatFormulaInner(inner)})</span>`);
  text = text.replace(/\\([a-zA-Z]+)/g, '<span class="math-op">$1</span>');
  return text;
}

const DECISION_LABELS = ["竞争解释", "预期证据", "诚实性声明", "观察", "推理", "假设", "动作", "理由", "判定", "诚实性"];

function decisionParts(text) {
  const source = String(text || "").trim();
  if (!source) return [];
  const marks = [];
  DECISION_LABELS.forEach((label) => {
    const token = `${label}：`;
    let from = 0;
    while (from < source.length) {
      const at = source.indexOf(token, from);
      if (at < 0) break;
      marks.push({ label, at, end: at + token.length });
      from = at + token.length;
    }
  });
  marks.sort((a, b) => a.at - b.at || b.label.length - a.label.length);
  const kept = [];
  marks.forEach((mark) => {
    const previous = kept[kept.length - 1];
    if (previous && mark.at < previous.end) return;
    kept.push(mark);
  });
  if (!kept.length) return [];
  return kept.map((mark, index) => ({
    label: mark.label,
    body: source.slice(mark.end, kept[index + 1] ? kept[index + 1].at : source.length).trim(),
  })).filter((part) => part.body);
}

function decisionCard(row) {
  const title = `${row.decision_id ? `${row.decision_id} · ` : ""}${row.action_zh || row.action || "规划受阻"}`;
  const text = row.reason_zh || row.detail || "";
  const parts = decisionParts(text);
  if (!parts.length) return `<article class="decision-card"><h4>${esc(title)}</h4><p class="agentic-prose">${renderProse(text)}</p></article>`;
  const lead = parts.find((part) => part.label === "观察") || parts[0];
  const rest = parts.filter((part) => part !== lead).map((part) => `<div><span>${esc(part.label)}</span><p>${renderProse(part.body)}</p></div>`).join("");
  return `<article class="decision-card"><h4>${esc(title)}</h4><p class="agentic-prose">${renderProse(lead.body)}</p>${rest ? `<div class="decision-fields">${rest}</div>` : ""}</article>`;
}

function agenticCard() {
  const rows = (state.agentic && state.agentic.campaigns) || [];
  if (!rows.length) return `<h2>代码级研究</h2><p class="muted">还没有代码级研究。用命令行 create 后在这里查看。</p>`;
  return rows.map((camp) => {
    const budget = camp.budget || {};
    const last = (camp.decisions || []).slice(-1)[0] || {};
    const best = (camp.experiments || []).filter((row) => row.evaluation_valid && row.candidate_id !== "baseline")
      .sort((a, b) => (b.delta_vs_control_pp ?? -1e9) - (a.delta_vs_control_pp ?? -1e9))[0];
    const progress = camp.live_progress ? `（epoch ${esc(camp.live_progress.epoch)} / ${esc(camp.live_progress.epochs)}）` : "";
    const expRows = (camp.experiments || []).map((row) => `<tr><td>${esc(row.candidate_id === "baseline" ? "对照" : row.candidate_id)}</td><td>${esc(row.fidelity === "pilot" ? "试跑" : "完整")}</td><td>${row.fixed_bank_top1 == null ? "未评估" : formatTick(row.fixed_bank_top1)}</td><td>${row.delta_vs_control_pp == null ? "—" : `${row.delta_vs_control_pp > 0 ? "+" : ""}${esc(row.delta_vs_control_pp)} pp`}</td><td>${row.evaluation_valid ? (row.candidate_id === "baseline" ? "对照" : "暂未确认") : `无效：${esc(row.reason || "")}`}</td></tr>`).join("");
    const candidates = (camp.candidates || []).map((row) => `<details data-keep="candidate-${esc(camp.campaign_id)}-${esc(row.candidate_id)}"><summary>${esc(row.candidate_id)} · ${esc(row.status)}</summary>${row.review_summary ? `<p class="agentic-prose">${renderProse(row.review_summary)}</p>` : ""}<pre class="agentic-source">${esc(row.source || "没有写出文件")}</pre></details>`).join("");
    const analyses = (camp.analyses || []).filter(Boolean).map((row) => `<p class="agentic-prose">${renderProse(row.summary_zh || "")}</p>`).join("");
    const decisions = (camp.decisions || []).slice().reverse().map(decisionCard).join("");
    const detail = camp.detail ? (AGENTIC_DETAIL[camp.detail] || camp.detail) : "";
    const gpu = budget.gpu_seconds_used == null ? "未记录" : Math.round(budget.gpu_seconds_used);
    const fee = budget.api_usd == null ? "未记录" : esc(budget.api_usd);
    const brief = firstClause(last.reason_zh || "");
    return `<h2>代码级研究 · ${esc(camp.objective || camp.campaign_id)}</h2>
      <p class="muted">${esc(camp.scope_zh || "")}。验证固定候选集 top1 用于比较，测试集本轮关闭。</p>
      <div class="agentic-summary">
        <div><span class="muted">当前问题</span><p>${esc(hypothesisLine(camp.hypothesis))}</p></div>
        <div><span class="muted">正在做什么</span><p>${esc(AGENTIC_STATUS[camp.status] || camp.status)}${progress}${detail ? `<span class="agentic-sub">${esc(detail)}</span>` : ""}</p></div>
        <div><span class="muted">最好可比结果</span><p>${best ? `${esc(best.candidate_id)} 相对对照 ${best.delta_vs_control_pp > 0 ? "+" : ""}${esc(best.delta_vs_control_pp)} pp<span class="agentic-sub">${best.fidelity === "pilot" ? "试跑" : "完整"}，暂未确认</span>` : "未评估"}</p></div>
        <div><span class="muted">剩余预算</span><ul class="agentic-budget"><li>训练 ${esc(budget.training_jobs ?? 0)}/${esc(budget.max_training_jobs ?? "—")}</li><li>模型调用 ${esc(budget.llm_calls ?? 0)}/${esc(budget.max_llm_calls ?? "—")}</li><li>GPU 秒 ${esc(gpu)}</li><li>费用 ${fee}</li></ul></div>
      </div>
      <p class="agentic-note">最近决定：${esc(last.action_zh || "—")}${brief ? `。${esc(brief)}` : ""}</p>
      <div class="trial-trace"><h3>实验</h3><div class="trial-scroll"><table><thead><tr><th>改动</th><th>阶段</th><th>开发指标</th><th>相对对照</th><th>证据状态</th></tr></thead><tbody>${expRows || `<tr><td colspan="5">还没有训练</td></tr>`}</tbody></table></div></div>
      ${analyses ? `<h3>结果分析</h3>${analyses}` : ""}
      ${candidates ? `<h3>候选代码</h3>${candidates}` : ""}
      <details data-keep="decisions-${esc(camp.campaign_id)}"><summary>决策记录</summary><div class="agentic-decisions">${decisions}</div></details>
      <div class="actions">
        <button type="button" data-agentic-action="resume" data-campaign="${esc(camp.campaign_id)}">继续</button>
        <button type="button" data-agentic-action="pause" data-campaign="${esc(camp.campaign_id)}">本轮后暂停</button>
        <button type="button" data-agentic-action="stop" data-campaign="${esc(camp.campaign_id)}">停止当前作业</button>
      </div>`;
  }).join("");
}

async function refreshAgentic() {
  try {
    const res = await fetch("/api/agentic_status", { cache: "no-store" });
    state.agentic = await res.json();
  } catch (_err) {
    return;
  }
  const box = document.getElementById("agentic-card");
  if (!box) return;
  const next = JSON.stringify(state.agentic);
  if (next === box.dataset.snapshot) return;
  const open = new Set();
  box.querySelectorAll("details[open][data-keep]").forEach((el) => open.add(el.dataset.keep));
  box.innerHTML = agenticCard();
  box.dataset.snapshot = next;
  box.querySelectorAll("details[data-keep]").forEach((el) => {
    if (open.has(el.dataset.keep)) el.open = true;
  });
  bindAgentic();
}

function bindAgentic() {
  document.querySelectorAll("[data-agentic-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.getAttribute("data-agentic-action");
      if (action === "stop" && !window.confirm("停止当前训练作业？已写入的结果会保留。")) return;
      const body = new URLSearchParams({ action, campaign: button.getAttribute("data-campaign") || "" });
      await fetch("/api/agentic_control", { method: "POST", body });
      refreshAgentic();
    });
  });
}

function scheduleAgentic() {
  if (state.agenticTimer) return;
  state.agenticTimer = window.setInterval(() => {
    if (state.surface === "retrieval") refreshAgentic();
  }, 5000);
}

async function renderRetrieval() {
  scheduleAgentic();
  refreshAgentic();
  paintRetrieval(state.shell || { discovered: false, campaigns: [] });
  const res = await fetch("/api/retrieval");
  const payload = await res.json();
  if (state.surface !== "retrieval" || state.discovered) return;
  state.shell = payload;
  paintRetrieval({ ...payload, discovered: false, subjects: [], gpus: [] });
}

function selectedGpuList() {
  return String(state.retrieval.gpu || "").split(",").map((item) => item.trim()).filter(Boolean);
}

function selectedSubjectList() {
  if (state.retrieval.subject === "all") return (state.subjects || []).map(String);
  return String(state.retrieval.subject || "").split(",").map((item) => item.trim()).filter((item) => item && item !== "all");
}

function splitBlock(title, rows) {
  const items = rows || [];
  if (!items.length) return "";
  return `<div class="choice"><span class="choice-label">${esc(title)}</span>${items.map((row) => `<p class="path">${esc(row.path)} · ${row.exists ? "存在" : "未找到"}</p>`).join("")}</div>`;
}

function subjectChecks(subjects) {
  if (!state.discovered) return `<p class="muted">点上方「发现数据与设备」后选择。</p>`;
  if (!subjects.length) return `<p class="muted">没有找到被试。</p>`;
  const selected = selectedSubjectList();
  const allOn = state.retrieval.subject === "all";
  return `<div class="gpu-checks" id="subject-checks">
    <label class="chip-check"><input type="checkbox" value="all"${allOn ? " checked" : ""}>全部</label>
    ${subjects.map((subject) => `<label class="chip-check"><input type="checkbox" value="${esc(subject)}"${allOn || selected.indexOf(String(subject)) >= 0 ? " checked" : ""}>${esc(subject)}</label>`).join("")}
  </div>`;
}

function gpuChecks(gpus) {
  if (!state.discovered) return `<p class="muted">点上方「发现数据与设备」后选择。</p>`;
  if (!gpus.length) return `<p class="muted">没有读到 GPU。</p>`;
  const auto = state.retrieval.gpu_mode === "auto_one";
  return `<label class="chip-check"><input type="checkbox" id="gpu-auto"${auto ? " checked" : ""}>自动选择一张空闲卡</label>
  <div class="gpu-checks" id="gpu-checks">
    ${gpus.map((row) => `<label class="chip-check"><input type="checkbox" value="${esc(row.index)}"${auto ? "" : selectedGpuList().indexOf(String(row.index)) >= 0 ? " checked" : ""}>${esc(row.index)} ${esc(row.name)} · 空闲 ${esc(row.memory_free_mb)} MB</label>`).join("")}
  </div>
  <p class="muted">自动选择只绑定一张空闲卡。也可以取消自动后勾选具体的卡。未勾选不会占用全部卡。</p>`;
}

const METRIC_NOTES = [
  ["fixed-bank Top-1", "固定候选集合中正确图像排第一的 query 比例。", "不跨候选集合比较。"],
  ["训练损失", "当前训练目标的优化值。", "不同目标或设置未必可比。"],
  ["GPU 小时", "本任务分配 GPU 的累计时长。", "不是美元费用或实际利用率。"],
];

function campaignFromHash() {
  const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
  return hash.startsWith("campaign/") ? hash.slice("campaign/".length) : "";
}

function paintRetrieval(payload) {
  const active = campaignFromHash();
  if (active) {
    paintProgress(active);
    return;
  }
  const options = payload.options || {};
  const datasets = options.datasets || [{ id: "eeg", label: "EEG" }, { id: "meg", label: "MEG" }];
  const protocols = options.protocols || [{ id: "intra-subject", label: "被试内" }, { id: "inter-subject", label: "被试间" }];
  const subjects = state.discovered ? (state.subjects || []) : [];
  const blockers = payload.blockers || [];
  const gpuCap = payload.gpu_seconds == null ? "未提供" : payload.gpu_seconds;
  const gpus = state.discovered ? (state.gpus || []) : [];
  const split = payload.split || {};
  const lrPlaceholder = payload.lr == null ? "" : payload.lr;
  const rootPlaceholder = payload.data_root || options.data_root || "";
  const openKeeps = new Set();
  const previousCard = document.getElementById("agentic-card");
  if (previousCard) previousCard.querySelectorAll("details[open][data-keep]").forEach((el) => openKeeps.add(el.dataset.keep));
  main.innerHTML = `
    <section class="research-page">
      <header>
        <h1>检索实验</h1>
        <p class="muted research-lead">选好数据和被试，点「开始研究」，agent 会规划并运行检索实验。</p>
        <details class="muted how-to"><summary>这个页面怎么用</summary>
          <p>先发现数据与设备，再开始研究。多被试合在一起训练、又没有留出被试时，验证是图像留出，不是跨被试泛化。固定候选 Top-1 才用于比较试验；训练 batch 里的 top1 只用来观察训练过程。GPU 小时是分配时长，不是费用。</p>
        </details>
      </header>
      <section class="card choice">
        <h2 class="step-title"><span class="step-no">1</span>数据</h2>
        <span class="choice-label">信号</span>
        <div class="segment" id="dataset-segment">
          ${datasets.map((row) => `<button type="button" data-dataset="${esc(row.id)}" class="${row.id === state.retrieval.dataset ? "active" : ""}">${esc(row.label)}</button>`).join("")}
        </div>
        <span class="choice-label">协议</span>
        <div class="segment" id="protocol-segment">
          ${protocols.map((row) => `<button type="button" data-protocol="${esc(row.id)}" class="${row.id === state.retrieval.exp_setting ? "active" : ""}">${esc(row.label)}</button>`).join("")}
        </div>
        <div class="discover-row">
          <label class="choice">数据根目录<input id="retrieval-root" spellcheck="false" value="${esc(state.retrieval.data_root)}" placeholder="${esc(rootPlaceholder)}"></label>
          <button type="button" id="retrieval-discover"${state.busy ? " disabled" : ""}>发现数据与设备</button>
        </div>
        <h2 class="step-title"><span class="step-no">2</span>被试与 GPU</h2>
        <span class="choice-label">被试</span>
        ${subjectChecks(subjects)}
        <span class="choice-label">GPU</span>
        ${gpuChecks(gpus)}
        <h2 class="step-title"><span class="step-no">3</span>研究方式</h2>
        <label class="choice">研究方式<select id="retrieval-policy">
          <option value="adaptive"${state.retrieval.policy === "adaptive" ? " selected" : ""}>自适应研究</option>
          <option value="legacy_fixed"${state.retrieval.policy === "legacy_fixed" ? " selected" : ""}>固定两试（对照）</option>
          <option value="agentic"${state.retrieval.policy === "agentic" ? " selected" : ""}>代码级研究</option>
        </select></label>
        ${state.retrieval.policy === "agentic"
          ? `<p class="muted">结束由研究循环决定，不使用两试早停。</p>`
          : `<label class="choice">结束条件<select id="retrieval-stop">${stopOptions(options)}</select></label>`}
        <details class="advanced-settings" id="advanced-settings"${state.advancedOpen ? " open" : ""}>
        <summary>高级设置</summary>
        <div class="advanced-body">
        <label class="choice">训练集目录（留空则自动）<input id="retrieval-train" spellcheck="false" value="${esc(state.retrieval.train_dir)}" placeholder="含 train.pt 的目录"></label>
        <label class="choice">测试集目录（留空则自动）<input id="retrieval-test" spellcheck="false" value="${esc(state.retrieval.test_dir)}" placeholder="含 test.pt 的目录"></label>
        <div class="budget-row">
          <label class="choice">epoch<input id="retrieval-epochs" inputmode="numeric" value="${esc(state.retrieval.epochs)}"></label>
          <label class="choice">seed<input id="retrieval-seed" inputmode="numeric" value="${esc(state.retrieval.seed)}"></label>
          <label class="choice">batch size<input id="retrieval-batch" inputmode="numeric" value="${esc(state.retrieval.batch_size)}"></label>
          <label class="choice">learning rate<input id="retrieval-lr" inputmode="decimal" value="${esc(state.retrieval.lr)}" placeholder="${esc(lrPlaceholder)}"></label>
          <label class="choice">GPU 秒数上限<input id="retrieval-gpu-seconds" inputmode="numeric" value="${esc(state.retrieval.gpu_seconds)}" placeholder="28800"></label>
        </div>
        <label class="choice">训练方式<select id="retrieval-strategy">
          <option value="pooled_subjects"${state.retrieval.training_strategy === "pooled_subjects" ? " selected" : ""}>多被试合训</option>
          <option value="per_subject"${state.retrieval.training_strategy === "per_subject" ? " selected" : ""}>每被试一个模型</option>
        </select></label>
        <label class="choice">泛化对象<select id="retrieval-generalization">
          <option value=""${state.retrieval.generalization_target === "" ? " selected" : ""}>按协议自动</option>
          <option value="seen_subject_unseen_stimulus"${state.retrieval.generalization_target === "seen_subject_unseen_stimulus" ? " selected" : ""}>见过的被试，未见图像</option>
          <option value="held_out_subject"${state.retrieval.generalization_target === "held_out_subject" ? " selected" : ""}>留出被试</option>
        </select></label>
        <label class="choice">留出被试<input id="retrieval-heldout" spellcheck="false" value="${esc(state.retrieval.held_out_subjects)}" placeholder="要求留出被试时填写，例如 sub-10"></label>
        </div>
        </details>
        <p class="protocol-line" id="protocol-line">当前划分：${esc(protocolPreview())}</p>
        <div class="research-actions action-bar">
          <details class="muted advanced"><summary>高级</summary>
            <div class="advanced-buttons">
              <button type="button" id="retrieval-probe"${state.busy ? " disabled" : ""}>预检</button>
              <button type="button" id="retrieval-dry"${state.busy ? " disabled" : ""}>试运行</button>
            </div>
          </details>
          <button type="button" class="primary" id="retrieval-run"${state.busy ? " disabled" : ""}>开始研究</button>
        </div>
        ${statusCard()}
      </section>
      <section class="research-note" id="retrieval-result">
        ${outcomeText(payload, blockers)}
      </section>
      <section class="card agentic-card" id="agentic-card">${agenticCard()}</section>
      <details class="card tech-details" id="retrieval-tech"${state.techOpen ? " open" : ""}>
        <summary>技术详情</summary>
        <div id="split-panel">
          <h2>划分</h2>
          <p>${esc(payload.protocol_label || protocolPreview())}</p>
          <p class="muted">${esc(split.note || "提交预检后显示训练、验证和测试文件。")}</p>
          ${splitBlock("训练文件", split.train_files)}
          ${splitBlock("验证文件", split.val_files)}
          ${splitBlock("不进入训练的测试文件", split.forbidden_files)}
          ${splitBlock("特征缓存", split.feature_caches)}
        </div>
        <p class="muted">测试结果单独显示，未用于选择。api_usd 保持空。GPU 秒数上限：${esc(gpuCap)}。学习率：${esc(payload.lr == null ? "未计算" : payload.lr)}。batch size：${esc(payload.batch_size || state.retrieval.batch_size)}</p>
        <h2>指标说明</h2>
        ${METRIC_NOTES.map((row) => `<p><b>${esc(row[0])}</b> ${esc(row[1])} ${esc(row[2])}</p>`).join("")}
        <h2>最近的 campaign</h2>
        <div class="campaign-list" id="campaign-list">
          ${(payload.campaigns || []).map((row) => `<button type="button" data-campaign="${esc(row.campaign_id)}">${esc(row.campaign_id)}</button>`).join("") || `<p class="muted">还没有检索 campaign</p>`}
        </div>
      </details>
    </section>`;
  bindRetrieval();
  bindAgentic();
  const agenticBox = document.getElementById("agentic-card");
  if (agenticBox) {
    agenticBox.dataset.snapshot = JSON.stringify(state.agentic || {});
    agenticBox.querySelectorAll("details[data-keep]").forEach((el) => {
      if (openKeeps.has(el.dataset.keep)) el.open = true;
    });
  }
  if (state.focusHeldout) {
    state.focusHeldout = false;
    const heldout = document.getElementById("retrieval-heldout");
    if (heldout) heldout.focus();
  }
  const consoleBox = document.getElementById("retrieval-console");
  if (consoleBox) consoleBox.scrollTop = consoleBox.scrollHeight;
}

function protocolPreview() {
  const r = state.retrieval;
  const target = r.generalization_target || (r.exp_setting === "intra-subject" || r.subject === "all" ? "seen_subject_unseen_stimulus" : "held_out_subject");
  const pooled = r.training_strategy === "per_subject" ? "每被试一个模型" : "多被试合训";
  if (target === "held_out_subject") return `${pooled} · 留出被试`;
  if (r.subject === "all" && r.training_strategy !== "per_subject") return "多被试合训 · 图像留出验证";
  return `${pooled} · 未见图像验证`;
}

function refreshProtocolLine() {
  const line = document.getElementById("protocol-line");
  if (line) line.textContent = "当前划分：" + protocolPreview();
}

function bindRetrieval() {
  const advanced = document.getElementById("advanced-settings");
  if (advanced) advanced.addEventListener("toggle", () => { state.advancedOpen = advanced.open; });
  const tech = document.getElementById("retrieval-tech");
  if (tech) tech.addEventListener("toggle", () => { state.techOpen = tech.open; });
  document.querySelectorAll("#dataset-segment button").forEach((button) => {
    button.addEventListener("click", () => {
      readRetrievalFields();
      state.retrieval.dataset = button.dataset.dataset;
      clearDiscovery();
      paintRetrieval(state.shell || {});
    });
  });
  document.querySelectorAll("#protocol-segment button").forEach((button) => {
    button.addEventListener("click", () => {
      state.retrieval.exp_setting = button.dataset.protocol;
      paintRetrieval(state.shell || {});
    });
  });
  ["retrieval-epochs", "retrieval-seed", "retrieval-batch", "retrieval-lr", "retrieval-gpu-seconds", "retrieval-stop", "retrieval-policy", "retrieval-strategy", "retrieval-generalization", "retrieval-heldout", "retrieval-root", "retrieval-train", "retrieval-test"].forEach((id) => {
    const node = document.getElementById(id);
    if (!node) return;
    node.addEventListener("change", () => {
      const previous = state.retrieval.data_root;
      readRetrievalFields();
      if (id === "retrieval-root" && state.retrieval.data_root !== previous) {
        clearDiscovery();
        paintRetrieval(state.shell || {});
      }
    });
  });
  const subjectChecks = document.getElementById("subject-checks");
  if (subjectChecks) subjectChecks.addEventListener("change", onSubjectChange);
  const gpuAuto = document.getElementById("gpu-auto");
  if (gpuAuto) gpuAuto.addEventListener("change", () => {
    state.retrieval.gpu_mode = gpuAuto.checked ? "auto_one" : "explicit";
    if (gpuAuto.checked) state.retrieval.gpu = "";
    readRetrievalFields();
  });
  const gpuChecks = document.getElementById("gpu-checks");
  if (gpuChecks) gpuChecks.addEventListener("change", () => {
    state.retrieval.gpu_mode = "explicit";
    const auto = document.getElementById("gpu-auto");
    if (auto) auto.checked = false;
    readRetrievalFields();
  });
  document.getElementById("retrieval-discover").addEventListener("click", () => submitRetrieval("discover"));
  document.getElementById("retrieval-probe").addEventListener("click", () => submitRetrieval("probe"));
  document.getElementById("retrieval-dry").addEventListener("click", () => submitRetrieval("dry_run"));
  document.getElementById("retrieval-run").addEventListener("click", () => submitRetrieval("run"));
  document.querySelectorAll("#campaign-list button").forEach((button) => {
    button.addEventListener("click", () => { location.hash = "campaign/" + encodeURIComponent(button.dataset.campaign); });
  });
}

function readRetrievalFields() {
  const fields = {
    "retrieval-epochs": "epochs",
    "retrieval-seed": "seed",
    "retrieval-batch": "batch_size",
    "retrieval-lr": "lr",
    "retrieval-gpu-seconds": "gpu_seconds",
    "retrieval-stop": "stop",
    "retrieval-policy": "policy",
    "retrieval-strategy": "training_strategy",
    "retrieval-generalization": "generalization_target",
    "retrieval-heldout": "held_out_subjects",
    "retrieval-root": "data_root",
    "retrieval-train": "train_dir",
    "retrieval-test": "test_dir",
  };
  Object.keys(fields).forEach((id) => {
    const node = document.getElementById(id);
    if (node) state.retrieval[fields[id]] = node.value;
  });
  const boxes = document.querySelectorAll("#gpu-checks input");
  if (boxes.length) {
    state.retrieval.gpu = [...boxes].filter((box) => box.checked).map((box) => box.value).join(",");
  }
  syncSubjectChecks();
  refreshProtocolLine();
}

function onSubjectChange(event) {
  const boxes = [...document.querySelectorAll("#subject-checks input")];
  const all = boxes.find((box) => box.value === "all");
  const names = boxes.filter((box) => box.value !== "all");
  if (event.target && event.target.value === "all") names.forEach((box) => { box.checked = all.checked; });
  else if (all) all.checked = names.length > 0 && names.every((box) => box.checked);
  syncSubjectChecks();
}

function syncSubjectChecks() {
  const boxes = [...document.querySelectorAll("#subject-checks input")];
  if (!boxes.length) return;
  const all = boxes.find((box) => box.value === "all");
  const names = boxes.filter((box) => box.value !== "all");
  state.retrieval.subject = all && all.checked ? "all" : names.filter((box) => box.checked).map((box) => box.value).join(",");
  refreshProtocolLine();
}

function humanBlocker(text) {
  const names = {
    no_subject: "还没有选择被试",
    gpu_not_selected: "没有选择 GPU。请勾选，或改成自动选择一张空闲卡。",
    held_out_subject_missing: "要求留出被试，但留出被试一栏是空的。",
  };
  return names[text] || String(text);
}

function actionLabel(action) {
  return { discover: "发现数据与设备", probe: "预检", dry_run: "试运行", run: "开始研究" }[action] || "等待操作";
}

function planLines(research) {
  if (!research) return [];
  if (research.action_label) {
    const lines = ["下一步：" + research.action_label];
    if (research.reason) lines.push(research.reason);
    return lines;
  }
  return [research.detail || "正在请求规划"];
}

function outcomeText(payload, blockers) {
  const action = state.retrievalAction;
  const lines = (blockers || []).map(humanBlocker);
  if (action === "dry_run") {
    const lead = payload.campaign_written ? "试运行已记录。validation top1 仍要一次真实训练才会出现。" : "试运行没有写入。";
    return `<p>${lead}</p>${lines.map((row) => `<p>${esc(row)}</p>`).join("")}`;
  }
  if (action === "run") {
    const research = payload.research;
    if (research && research.status === "planned") {
      const lead = payload.started ? "训练已启动。" : "这一步没有启动训练。";
      return `<p>${lead}</p>${planLines(research).map((row) => `<p>${esc(row)}</p>`).join("")}`;
    }
    return `<p>${payload.agentic_campaign ? `代码级研究已启动：${esc(payload.agentic_campaign)}` : (payload.started ? "训练已启动。" : "研究没有启动训练。")}</p>${lines.map((row) => `<p>${esc(row)}</p>`).join("")}`;
  }
  if (action === "discover") return `<p>检索完成。勾选被试和 GPU 后再试运行。</p>`;
  return lines.map((row) => `<p>${esc(row)}</p>`).join("") || `<p>预检通过。validation top1 仍要一次真实训练才会出现。</p>`;
}

function trainPhase() {
  return (state.trainStatus && state.trainStatus.phase) || "";
}

function planBlock() {
  const research = state.trainStatus && state.trainStatus.research;
  if (!research) return `<p class="muted" id="plan-line">规划还没有开始。</p>`;
  const lines = planLines(research).map((row) => `<p>${esc(row)}</p>`).join("");
  return `<div id="plan-line">${lines}</div>`;
}

function planRaw() {
  const research = state.trainStatus && state.trainStatus.research;
  if (!research || !research.raw) return "";
  return `<pre>${esc(JSON.stringify(research.raw))}</pre>`;
}

function progressView() {
  const phase = trainPhase();
  const status = state.trainStatus || {};
  const research = status.research || {};
  const epochs = Number(status.epochs) || 0;
  const epoch = Number(status.epoch) || 0;
  if (status.prior_curve && research.action_label) return { label: "下一步：" + research.action_label, width: 0, busy: false };
  if (status.prior_curve && !research.action_label) return { label: research.detail || "正在请求规划", width: 0, busy: research.status === "planning" };
  if (phase === "planning") return { label: "正在请求规划", width: 0, busy: true };
  if (phase === "loading") return { label: "正在加载数据", width: 0, busy: true };
  if (phase === "training" && epochs > 0) return { label: `训练中 ${epoch} / ${epochs}`, width: Math.round((epoch / epochs) * 100), busy: false };
  if (phase === "finished" && epochs > 0 && epoch < epochs) return { label: `已完成 · 提前停止，训练 ${epoch} 轮`, width: Math.round((epoch / epochs) * 100), busy: false };
  if (phase === "finished" && epochs > 0) return { label: `训练结束 ${epoch} / ${epochs}`, width: Math.round((epoch / epochs) * 100), busy: false };
  if (phase === "failed") return { label: "训练已停止", width: epochs > 0 ? Math.round((epoch / epochs) * 100) : 0, busy: false };
  const progress = state.progress;
  const known = progress && progress.total > 0;
  if (state.busy) return { label: "正在检查数据", width: 0, busy: true };
  if (known) return { label: `已找到 ${progress.found} / ${progress.total}`, width: Math.round((progress.found / progress.total) * 100), busy: false };
  return { label: "等待操作", width: 0, busy: false };
}

function stepMode(name) {
  const phase = trainPhase();
  const settled = state.phase === "done";
  if (name === "提交") return state.phase === "idle" ? "" : "done";
  if (name === "加载") {
    if (phase === "training" || phase === "finished" || phase === "failed") return "done";
    if (phase === "loading" || (state.busy && !phase)) return "active";
    if (settled) return "done";
    return "";
  }
  if (name === "训练") {
    if (phase === "finished") return "done";
    if (phase === "training" || phase === "failed") return "active";
    return "";
  }
  return settled ? "done" : "";
}

function stopOptions(options) {
  const rows = (options && options.stops) || [
    { id: "single_full", label: "单次跑满 epoch" },
    { id: "single_early", label: "单次验证集早停" },
    { id: "chain_full", label: "自动两试，每次跑满 epoch" },
    { id: "chain_early", label: "自动两试，验证集早停" },
  ];
  return rows.map((row) => `<option value="${esc(row.id)}"${row.id === state.retrieval.stop ? " selected" : ""}>${esc(row.label)}</option>`).join("");
}

function formatTick(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  const abs = Math.abs(number);
  if (abs >= 10) return number.toFixed(1);
  return number.toFixed(3);
}

function formatSetting(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number === 0) return "0";
  return String(Number(number.toPrecision(6)));
}

function axisChart(series, minY, maxY) {
  const width = 320;
  const height = 132;
  const left = 42;
  const right = 8;
  const top = 10;
  const bottom = 22;
  const count = series[0].values.length;
  const span = maxY - minY || 1;
  const xOf = (index) => left + (count === 1 ? (width - left - right) / 2 : index * (width - left - right) / (count - 1));
  const yOf = (value) => top + (1 - (value - minY) / span) * (height - top - bottom);
  const marks = series.map((item) => {
    if (count === 1) return `<circle cx="${xOf(0).toFixed(1)}" cy="${yOf(item.values[0]).toFixed(1)}" r="3.5" fill="${item.color}"></circle>`;
    const points = item.values.map((value, index) => `${xOf(index).toFixed(1)},${yOf(value).toFixed(1)}`).join(" ");
    return `<polyline fill="none" stroke="${item.color}" stroke-width="2" points="${points}"></polyline>`;
  }).join("");
  const yTicks = [maxY, (maxY + minY) / 2, minY].map((value) => `<text x="2" y="${(yOf(value) + 3).toFixed(1)}" fill="#64748b" font-size="10">${formatTick(value)}</text>`).join("");
  const first = series[0].epochs[0];
  const last = series[0].epochs[count - 1];
  const xTicks = `<text x="${xOf(0).toFixed(1)}" y="${height - 4}" fill="#64748b" font-size="10" text-anchor="${count === 1 ? "middle" : "start"}">${first}</text>${count > 1 ? `<text x="${xOf(count - 1).toFixed(1)}" y="${height - 4}" fill="#64748b" font-size="10" text-anchor="end">${last}</text>` : ""}`;
  return `<svg viewBox="0 0 ${width} ${height}" role="img"><line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#cbd5e1"></line><line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#cbd5e1"></line>${yTicks}${xTicks}${marks}</svg>`;
}

function lossChart(rows) {
  const values = rows.map((row) => Number(row.train_loss));
  const epochs = rows.map((row) => Number(row.epoch));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = low === high ? Math.max(Math.abs(low) * 0.05, 0.01) : (high - low) * 0.1;
  return `<div class="chart"><h3>训练 loss</h3>${axisChart([{ values, epochs, color: cssVar("--accent", "#0e7490") }], low - pad, high + pad)}</div>`;
}

function validationChart(rows) {
  const epochs = rows.map((row) => Number(row.epoch));
  const fixed = rows.every((row) => row.fixed_bank_top1 !== undefined);
  const top1 = fixed ? "fixed_bank_top1" : "val_top1";
  const top5 = fixed ? "fixed_bank_top5" : "val_top5";
  const title = fixed ? "validation 固定候选集" : "validation 批内（仅诊断）";
  return `<div class="chart"><h3>${title}</h3>${axisChart([
    { values: rows.map((row) => Number(row[top1])), epochs, color: cssVar("--accent", "#0e7490") },
    { values: rows.map((row) => Number(row[top5])), epochs, color: cssVar("--chart-top5", "#15803d") },
  ], 0, 1)}<div class="legend"><span class="swatch top1">top1</span><span class="swatch top5">top5</span></div></div>`;
}

const TRIAL_STATUS = { running: "训练中", finished: "已完成", failed: "失败", interrupted: "中断", unknown: "未知" };

function settingText(setting) {
  if (!setting || typeof setting !== "object") return String(setting || "设置未记录");
  return `seed ${setting.seed} · lr ${formatSetting(setting.learning_rate)} · weight decay ${formatSetting(setting.weight_decay)}`;
}

function trialTable() {
  const trials = (state.trainStatus && state.trainStatus.trials) || [];
  if (!trials.length) return "";
  const rows = trials.map((trial) => {
    const score = trial.fixed_bank_top1 === null || trial.fixed_bank_top1 === undefined
      ? (trial.duplicate_of ? `同 ${esc(trial.duplicate_of)}` : (trial.score_error ? "补算失败" : "待补算"))
      : formatTick(trial.fixed_bank_top1);
    const test = trial.test_result && trial.test_result.top1 !== undefined ? formatTick(trial.test_result.top1) : "—";
    return `<tr><td>${esc(trial.trial)}</td><td>${esc(settingText(trial.setting))}</td><td>${esc(TRIAL_STATUS[trial.status] || trial.status)}</td><td>${score}</td><td>${test}</td></tr>`;
  }).join("");
  return `<div class="trial-trace"><h3>试验轨迹</h3><div class="trial-scroll"><table><thead><tr><th>试验</th><th>设置</th><th>状态</th><th>验证固定候选集 top1</th><th>测试 top1（未用于选择）</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

function chainStrip() {
  const chain = state.trainStatus && state.trainStatus.chain;
  if (!chain || !chain.trials) return "";
  const rows = chain.trials.map((trial) => `<p>${esc(trial.trial)} · ${esc(trial.profile)} · weight decay ${esc(trial.weight_decay)} · ${esc(trial.action)}</p>`).join("");
  return `<div class="chart"><h3>${esc(chain.label || "试验")}</h3>${rows}<p class="muted">下一次是否开始只看 validation top1。</p></div>`;
}

function testLine() {
  const result = state.trainStatus && state.trainStatus.test_result;
  if (!result) return `<p class="muted">测试结果未写入。未用于选择。</p>`;
  return `<p>test top1 ${formatTick(result.top1)} · top5 ${formatTick(result.top5)}。未用于选择。</p>`;
}

function curveBlock() {
  const rows = (state.trainStatus && state.trainStatus.history) || [];
  if (!rows.length) return `<p class="muted">还没有 epoch</p>`;
  const prior = state.trainStatus && state.trainStatus.prior_curve ? `<p class="muted">已有训练记录</p>` : "";
  return `<div class="charts">${prior}${lossChart(rows)}${validationChart(rows)}</div>`;
}

function statusCard() {
  const view = progressView();
  const step = (name) => `<li class="${stepMode(name)}">${name}</li>`;
  const lines = state.consoleLines.length ? state.consoleLines.map((line) => esc(line)).join("\n") : "等待操作";
  const warn = state.refused || trainPhase() === "failed" ? " warn" : "";
  return `<div class="status-card" id="status-card">
    <div class="progress-head"><span>${esc(view.label)}</span></div>
    <div class="progress-track${view.busy ? " busy" : ""}${warn}"><div class="progress-fill" style="width:${view.busy ? 40 : view.width}%"></div></div>
    <ol class="steps">
      ${step("提交")}
      ${step("加载")}
      ${step("训练")}
      ${step("结果")}
    </ol>
    ${planBlock()}
    ${trialTable()}
    ${chainStrip()}
    ${testLine()}
    ${curveBlock()}
    <details class="muted"><summary>技术详情</summary>
      <p class="path">${esc(campaignFromHash() || campaignId())}</p>
      <div class="term">
        <div class="term-bar"><span>输出</span><span>${esc(actionLabel(state.retrievalAction))}</span></div>
        <pre id="retrieval-console">${lines}</pre>
        ${planRaw()}
      </div>
    </details>
  </div>`;
}

function pushConsole(line) {
  state.consoleLines.push(line);
  if (state.consoleLines.length > 80) state.consoleLines.splice(0, state.consoleLines.length - 80);
}

function campaignId() {
  const setting = String(state.retrieval.exp_setting || "").replace(/-/g, "_");
  const slug = String(state.retrieval.subject || "none").replace(/,/g, "_");
  return `${state.retrieval.dataset}_${setting}_${slug}_s${state.retrieval.seed || 0}`;
}

function refreshStatusCard() {
  const node = document.getElementById("status-card");
  if (!node) return;
  const oldTrack = node.querySelector(".progress-track");
  node.outerHTML = statusCard();
  const fresh = document.getElementById("status-card");
  const newTrack = fresh && fresh.querySelector(".progress-track");
  if (oldTrack && newTrack && oldTrack.classList.contains("busy") && newTrack.classList.contains("busy")) {
    oldTrack.className = newTrack.className;
    newTrack.replaceWith(oldTrack);
  }
  const planLine = document.getElementById("plan-line");
  if (state.planFresh && planLine) {
    planLine.classList.add("plan-fresh");
    state.planFresh = false;
  }
  const consoleBox = document.getElementById("retrieval-console");
  if (consoleBox) consoleBox.scrollTop = consoleBox.scrollHeight;
}

function stopTrainPoll() {
  if (state.trainPoll) clearInterval(state.trainPoll);
  state.trainPoll = null;
}

async function pollTrainStatus() {
  try {
    const res = await fetch("/api/train_status?campaign=" + encodeURIComponent(campaignId()));
    if (!res.ok) return;
    state.trainStatus = await res.json();
    const research = state.trainStatus && state.trainStatus.research;
    if (research) {
      const stamp = [research.detail, research.action_label, research.reason].join("|");
      if (stamp !== state.planStamp) {
        state.planStamp = stamp;
        state.planFresh = true;
        planLines(research).forEach(pushConsole);
      }
    }
    refreshStatusCard();
  } catch (error) {
    return;
  }
}

function startTrainPoll() {
  stopTrainPoll();
  state.trainStatus = null;
  pollTrainStatus();
  state.trainPoll = setInterval(pollTrainStatus, 1000);
}

async function submitRetrieval(action) {
  if (state.busy) return;
  readRetrievalFields();
  state.busy = true;
  state.phase = "check";
  state.refused = false;
  state.retrievalAction = action;
  pushConsole("提交" + actionLabel(action));
  paintRetrieval(state.shell || {});
  if (action === "run" && state.retrieval.policy !== "agentic") startTrainPoll();
  let payload = state.shell || {};
  try {
    const res = await fetch("/api/retrieval", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: retrievalBody(action),
    });
    payload = await res.json();
    const rows = payload.log && payload.log.length ? payload.log : (payload.blockers || ["请求失败"]).map(humanBlocker);
    rows.forEach(pushConsole);
    state.progress = payload.progress || null;
    const planned = payload.research && payload.research.status === "planned";
    state.refused = Boolean(payload.research && payload.research.status === "planning_blocked") || (action === "run" && payload.ok === false && !planned);
    if (payload.research) {
      state.trainStatus = Object.assign({}, state.trainStatus || {}, { research: payload.research, phase: payload.research.phase || trainPhase() });
    }
    if (payload.discovered) {
      state.discovered = true;
      state.subjects = payload.subjects || [];
      state.gpus = payload.gpus || [];
    }
    if ((payload.blockers || []).includes("held_out_subject_missing")) {
      state.advancedOpen = true;
      state.focusHeldout = true;
    }
    state.shell = payload;
  } catch (error) {
    pushConsole("请求失败");
    state.refused = true;
  }
  if (action === "run" && state.retrieval.policy !== "agentic") await pollTrainStatus();
  if (payload && payload.agentic_campaign) {
    agenticRequestId = "";
    refreshAgentic();
  }
  stopTrainPoll();
  state.busy = false;
  state.phase = "done";
  if (state.surface !== "retrieval") return;
  paintRetrieval(payload);
}

function paintProgress(campaign) {
  main.innerHTML = `
    <section class="research-page">
      <header>
        <h1>研究进行</h1>
        <p class="muted">这一页只看当前 campaign。创建新研究请返回。</p>
        <p><a href="#" id="back-create">返回创建</a></p>
      </header>
      ${statusCard()}
    </section>`;
  document.getElementById("back-create").addEventListener("click", (event) => {
    event.preventDefault();
    location.hash = "";
  });
  fetch("/api/train_status?campaign=" + encodeURIComponent(campaign))
    .then((res) => res.ok ? res.json() : null)
    .then((payload) => {
      if (campaignFromHash() !== campaign || !payload) return;
      state.trainStatus = payload;
      const node = document.getElementById("status-card");
      if (node) node.outerHTML = statusCard();
    })
    .catch(() => {});
}

async function showCampaign(campaignId) {
  const box = document.getElementById("retrieval-result");
  if (!box) return;
  const res = await fetch("/api/research?campaign=" + encodeURIComponent(campaignId));
  if (!res.ok) {
    box.innerHTML = `<p>campaign 无法读取</p>`;
    return;
  }
  const payload = await res.json();
  const comparison = payload.comparison || {};
  const testResult = comparison.test_result == null ? "空" : comparison.test_result;
  box.innerHTML = `<p>${esc(campaignId)}</p><p>test_result：${esc(testResult)}</p><p>分数不进入数值检查卡片。</p>`;
}

window.addEventListener("hashchange", () => {
  if (state.surface === "retrieval") paintRetrieval(state.shell || {});
});
document.getElementById("surface-home").addEventListener("click", () => setSurface("home"));
document.getElementById("surface-screen").addEventListener("click", () => setSurface("screen"));
document.getElementById("surface-retrieval").addEventListener("click", () => setSurface("retrieval"));
setSurface(campaignFromHash() ? "retrieval" : "home");
