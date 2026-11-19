"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const state = { runs: [], runId: null, detail: null, offset: 0, limit: 30, status: "", defect: "", query: "" };
  const LABELS = { "不通过": "已确认问题", "待判定": "候选待核查", "通过": "通过", "不适用": "不适用", confirmed: "已确认问题", candidate: "候选待核查" };
  const statusClass = (value) => value === "不通过" || value === "confirmed" ? "fail" : value === "通过" ? "pass" : value === "待判定" || value === "candidate" ? "pending" : "";
  const displayStatus = (value) => LABELS[value] || value || "未记录";
  const fmt = (value) => value === null || value === undefined || value === "" ? "—" : Number(value).toLocaleString("zh-CN");
  const api = async (path) => {
    const response = await fetch(path, { headers: { Accept: "application/json" }, credentials: "same-origin" });
    let data;
    try { data = await response.json(); } catch (_) { throw new Error(`服务器返回了无法读取的内容（HTTP ${response.status}）。`); }
    if (!response.ok) throw new Error(data?.error || `请求失败（HTTP ${response.status}）。`);
    return data;
  };
  const safeJson = (value) => { try { return JSON.stringify(value, null, 2); } catch (_) { return String(value ?? ""); } };
  const issueText = (finding) => finding?.description || finding?.message || finding?.summary || finding?.title || finding?.basis || finding?.detail || "打开详情查看判断依据";
  const locationText = (finding) => {
    const locator = finding?.locator || finding?.location || finding?.evidenceLocator;
    if (!locator) return "样本级记录";
    if (typeof locator === "string") return locator;
    if (Array.isArray(locator.bbox)) return `区域框 [${locator.bbox.join(", ")}]`;
    if (Array.isArray(locator.polygon)) return "多边形区域";
    if (locator.annotation_pointer || locator.annotationPointer) return locator.annotation_pointer || locator.annotationPointer;
    if (locator.timeInterval || locator.time_interval || locator.interval) return "时间区间 " + safeJson(locator.timeInterval || locator.time_interval || locator.interval);
    return locator.type || "已记录定位";
  };
  function showError(message) {
    $("review-main").innerHTML = `<div class="review-error"><strong>无法读取案例工作台</strong><p>${esc(message)}</p><button id="review-retry" class="button secondary small" type="button">重新读取</button></div>`;
    $("review-retry").addEventListener("click", loadRuns);
  }
  function renderRuns() {
    const list = $("run-list"); $("run-count").textContent = `${state.runs.length} 个`; list.replaceChildren();
    if (!state.runs.length) { list.innerHTML = `<div class="run-empty">当前服务没有发现历史评价。请确认服务器启动时配置了 <code>--run-root</code>，或先在“数据控制台”完成一次评价。</div>`; return; }
    for (const run of state.runs) {
      const button = document.createElement("button"); button.type = "button"; button.className = `run-button${run.id === state.runId ? " selected" : ""}`;
      button.innerHTML = `<strong>${esc(run.name || run.runId || run.id)}</strong><small>${esc(run.runId || run.id)} · ${fmt(run.findingCount)} 条记录</small>`;
      button.addEventListener("click", () => { state.runId = run.id; state.offset = 0; renderRuns(); loadRun().catch((error) => showError(error.message)); }); list.append(button);
    }
  }
  function stats(detail, visible) {
    const counts = detail?.statusCounts || {};
    const confirmed = Number(counts["不通过"] || counts.confirmed || 0); const candidates = Number(counts["待判定"] || counts.candidate || 0); const verified = visible.filter((f) => f._verified === true).length;
    return `<div class="review-stats"><div class="review-stat"><span>当前运行问题记录</span><strong>${fmt(detail?.findingCount)}</strong></div><div class="review-stat teal"><span>已确认问题</span><strong>${fmt(confirmed)}</strong></div><div class="review-stat"><span>候选待核查</span><strong>${fmt(candidates)}</strong></div><div class="review-stat"><span>本页证据已校验</span><strong>${fmt(verified)}</strong></div></div>`;
  }
  function toolbar(detail) {
    const defects = Object.keys(detail?.defectCounts || {}).filter(Boolean).sort();
    return `<div class="review-toolbar"><div class="field"><label for="review-status">状态</label><select id="review-status"><option value="">全部状态</option><option value="不通过">已确认问题</option><option value="待判定">候选待核查</option><option value="通过">通过</option></select></div><div class="field"><label for="review-defect">缺陷类型</label><select id="review-defect"><option value="">全部类型</option>${defects.map((x) => `<option value="${esc(x)}">${esc(x)}</option>`).join("")}</select></div><div class="field search"><label for="review-query">搜索样本、依据或检查编号</label><input id="review-query" type="search" maxlength="120" placeholder="例如样本名、L3-06、尺寸矛盾"></div><button id="review-refresh-run" type="button" class="button secondary">刷新案例</button></div>`;
  }
  function table(page) {
    if (!page.items?.length) return `<div class="review-loading">当前筛选没有案例。可以清空筛选条件或切换左侧运行。</div>`;
    const rows = page.items.map((finding) => {
      const status = displayStatus(finding.status); const sample = finding.sampleId || finding.dataName || finding.labelName || finding.datasetId || finding.findingId; const id = finding.findingId || "";
      return `<tr><td><button class="case-button" type="button" data-finding="${esc(id)}">${esc(sample)}<small>${esc(id)}</small></button></td><td><span class="pill ${statusClass(finding.status)}">${esc(status)}</span><br><small>${esc(finding.defectType || "未分类")}</small></td><td class="basis">${esc(issueText(finding))}</td><td class="location">${esc(locationText(finding))}</td><td><button class="button secondary small" type="button" data-finding="${esc(id)}">打开证据 ↗</button></td></tr>`;
    }).join("");
    return `<div class="review-table-wrap"><table class="review-table"><thead><tr><th>样本 / 记录</th><th>状态 / 类型</th><th>判断依据</th><th>定位</th><th>操作</th></tr></thead><tbody>${rows}</tbody></table></div><div class="review-pagination"><span>显示 ${page.total ? page.offset + 1 : 0}–${page.offset + page.items.length} / ${fmt(page.total)} 条</span><div class="review-pagination-actions"><button id="review-prev" class="button secondary small" type="button" ${page.offset <= 0 ? "disabled" : ""}>上一页</button><button id="review-next" class="button secondary small" type="button" ${page.offset + page.items.length >= page.total ? "disabled" : ""}>下一页</button></div></div>`;
  }
  function bindMain(detail) {
    $("review-status").value = state.status; $("review-defect").value = state.defect; $("review-query").value = state.query;
    $("review-status").addEventListener("change", () => { state.status = $("review-status").value; state.offset = 0; loadFindings().catch((error) => showError(error.message)); });
    $("review-defect").addEventListener("change", () => { state.defect = $("review-defect").value; state.offset = 0; loadFindings().catch((error) => showError(error.message)); });
    $("review-query").addEventListener("keydown", (event) => { if (event.key === "Enter") { state.query = $("review-query").value.trim(); state.offset = 0; loadFindings().catch((error) => showError(error.message)); } });
    $("review-refresh-run").addEventListener("click", () => loadRun().catch((error) => showError(error.message)));
    $("review-prev")?.addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadFindings().catch((error) => showError(error.message)); });
    $("review-next")?.addEventListener("click", () => { state.offset += state.limit; loadFindings().catch((error) => showError(error.message)); });
    document.querySelectorAll("[data-finding]").forEach((button) => button.addEventListener("click", () => openFinding(button.dataset.finding)));
  }
  async function loadFindings() {
    if (!state.runId) return;
    const params = new URLSearchParams({ limit: String(state.limit), offset: String(state.offset) }); if (state.status) params.set("status", state.status); if (state.defect) params.set("defect", state.defect); if (state.query) params.set("q", state.query);
    const page = await api(`/api/runs/${encodeURIComponent(state.runId)}/findings?${params}`); const detail = state.detail; const main = $("review-main");
    main.innerHTML = `${toolbar(detail)}${stats(detail, page.items || [])}<div class="review-note"><strong>核查提示：</strong>“已确认问题”来自当前保存的检查证据；“候选待核查”是模型或先验给出的调查线索。点击样本即可查看证据链，页面不会修改原始结果。</div>${table(page)}`; bindMain(detail);
  }
  async function loadRun() { if (!state.runId) return; $("review-main").innerHTML = `<div class="review-loading"><span class="spinner"></span>正在读取运行记录与案例…</div>`; state.detail = await api(`/api/runs/${encodeURIComponent(state.runId)}`); state.status = ""; state.defect = ""; state.query = ""; await loadFindings(); }
  async function loadRuns() { try { const result = await api("/api/runs?limit=200"); state.runs = Array.isArray(result.items) ? result.items : []; if (!state.runId || !state.runs.some((run) => run.id === state.runId)) state.runId = state.runs[0]?.id || null; renderRuns(); if (state.runId) await loadRun(); else $("review-main").innerHTML = `<div class="review-loading">请选择左侧运行记录。</div>`; } catch (error) { showError(error.message); } }
  async function openFinding(findingId) {
    if (!findingId || !state.runId) return; $("review-modal").hidden = false; $("review-dialog-title").textContent = "正在读取案例…"; $("review-dialog-body").innerHTML = `<div class="review-loading"><span class="spinner"></span>正在读取证据与原始样本…</div>`;
    try {
      const payload = await api(`/api/runs/${encodeURIComponent(state.runId)}/evidence/${encodeURIComponent(findingId)}`); const finding = payload.finding || {}; const evidence = payload.evidence || {}; const sample = payload.sample; let sampleDetail = null;
      if (sample?.url) { try { sampleDetail = await api(sample.url); } catch (_) { /* Evidence remains useful when source preview is unavailable. */ } }
      $("review-dialog-kicker").textContent = `${finding.defectType || "问题记录"} · ${displayStatus(finding.status)}`; $("review-dialog-title").textContent = finding.sampleId || finding.dataName || finding.findingId || "案例详情";
      const image = sampleDetail?.imageUrl; const visual = image ? `<div class="review-image-stage"><img src="${esc(image)}" alt="原始样本预览"></div>` : `<div class="review-image-stage"><div class="no-image">当前记录没有可直接预览的 PNG。<br>${sample ? "仍可通过证据、文件名和原始 JSON 回查。" : "证据没有唯一映射到原始样本。"}</div></div>`;
      const labels = [["结论", issueText(finding)], ["状态", displayStatus(finding.status)], ["定位", locationText(finding)], ["证据完整性", payload.sourceVerified ? "已通过保存证据 SHA 与身份字段校验" : "未通过或未记录完整校验，需人工复核"], ["回查样本", sample ? `${sample.dataName || sample.name || "已找到样本"}（${sample.matchMethod || "记录映射"}）` : "当前记录未能唯一映射原始样本"]];
      const sections = labels.map(([title, value]) => `<section class="review-detail-section"><h3>${esc(title)}</h3><p>${esc(value)}</p></section>`).join("");
      $("review-dialog-body").innerHTML = `<div class="review-dialog-grid"><div>${visual}<p class="section-note">原始标注叠加仍以“数据控制台”样本预览为准；这里展示的是当前案例的证据入口。</p>${sampleDetail ? `<details><summary>原始标注 JSON</summary><pre>${esc(safeJson(sampleDetail.label))}</pre></details>` : ""}</div><div class="review-detail-sections">${sections}<section class="review-detail-section"><h3>工具观察与测量</h3><pre>${esc(safeJson(evidence.observation ?? evidence.toolOutput ?? evidence.observed ?? evidence))}</pre></section><section class="review-detail-section"><h3>完整问题记录</h3><pre>${esc(safeJson(finding))}</pre></section></div></div>`;
    } catch (error) {
      $("review-dialog-title").textContent = "读取失败"; $("review-dialog-body").innerHTML = `<div class="review-error"><strong>案例详情读取失败</strong><p>${esc(error.message)}</p><button id="review-detail-retry" class="button secondary small" type="button">重新读取</button></div>`; $("review-detail-retry").addEventListener("click", () => openFinding(findingId));
    }
  }
  function closeModal() { $("review-modal").hidden = true; }
  $("review-dialog-close").addEventListener("click", closeModal); $("review-dialog-close-bottom").addEventListener("click", closeModal); $("review-modal").addEventListener("click", (event) => { if (event.target === $("review-modal")) closeModal(); }); document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeModal(); }); $("refresh-review").addEventListener("click", () => loadRuns()); $("review-language").addEventListener("click", () => { const button = $("review-language"); button.textContent = button.textContent === "EN" ? "中文" : "EN"; }); loadRuns();
})();
