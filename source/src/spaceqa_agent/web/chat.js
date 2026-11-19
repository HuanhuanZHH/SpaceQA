"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const TERMINAL = new Set(["completed", "partial", "failed", "succeeded", "cancelled"]);
  const LABELS = { queued: "等待分析", running: "正在分析", completed: "分析完成", succeeded: "已完成", partial: "部分完成", failed: "分析失败", pending: "等待执行", skipped: "已跳过", cancelled: "已取消", pass: "已通过" };
  const state = { sessions: [], folders: [], folderPath: "", selectedDataset: null, folderToken: 0, folderLoaded: false, folderLoading: false, demo: null, activeId: null, current: null, viewToken: 0, revision: new Map(), sending: new Map(), pollTimer: null, readController: null, creating: false, upload: null, noticeAction: null, renderedKey: "", listToken: 0 };
  const storageKey = "spaceqa.chat.activeSession";
  const DEFECTS = [ ["L0-01", "数据文件缺失"], ["L2-G", "边界与位置偏差"], ["L2-P", "类别标注错误"], ["L2-T", "起止时刻抖动"], ["L2-F", "元数据尺寸矛盾"], ["L3-06", "漏标"], ["L3-07", "虚标"], ["L3-MIS", "覆盖不足或过度"] ];
  const workspace = { view: "overview", key: "", insights: null, insightController: null, casesController: null, tierController: null, detailController: null, explainController: null, casesToken: 0, tierToken: 0, detailToken: 0, offset: 0, limit: 12, total: 0, tier: "", detail: null, searchTimer: null };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text).replaceAll("待评估", "需补充依据");
    return node;
  }
  function showNotice(message, action, actionLabel = "重试", error = true) {
    $("notice-text").textContent = message;
    $("notice").classList.toggle("error", error);
    $("notice").hidden = false;
    state.noticeAction = action || null;
    $("notice-action").hidden = !action;
    $("notice-action").textContent = actionLabel;
  }
  function hideNotice() { $("notice").hidden = true; state.noticeAction = null; }
  function connection(online) {
    $("connection-dot").className = `connection-dot ${online ? "online" : "offline"}`;
    $("connection-label").textContent = online ? "已连接服务器" : "连接中断";
  }
  function errorMessage(error) {
    if (error?.name === "AbortError") return "请求超时，服务器可能仍在处理。请刷新会话确认。";
    return error?.message || "无法连接服务器，请检查连接后重试。";
  }
  async function request(url, options = {}) {
    const { timeout = 20000, ...rest } = options;
    const controller = new AbortController();
    const external = rest.signal;
    const abort = () => controller.abort();
    if (external?.aborted) controller.abort();
    else external?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(abort, timeout);
    try {
      const response = await fetch(url, { credentials: "same-origin", ...rest, signal: controller.signal, headers: { Accept: "application/json", ...(rest.body ? { "Content-Type": "application/json" } : {}), ...(rest.headers || {}) } });
      let result;
      try { result = await response.json(); } catch (_) { throw new Error(`服务器返回了无法读取的内容（HTTP ${response.status}）。`); }
      if (!response.ok) throw new Error(result?.error?.message || result?.message || (typeof result?.error === "string" ? result.error : `请求失败（HTTP ${response.status}）。`));
      connection(true);
      return result;
    } catch (error) {
      if (error instanceof TypeError) { connection(false); throw new Error("无法连接服务器，请检查服务或本机连接后重试。"); }
      throw error;
    } finally { clearTimeout(timer); external?.removeEventListener("abort", abort); }
  }
  const post = (url, data, timeout) => request(url, { method: "POST", body: JSON.stringify(data), timeout });
  function remember(id) { try { id ? localStorage.setItem(storageKey, id) : localStorage.removeItem(storageKey); } catch (_) { /* Storage may be disabled. */ } }
  function remembered() { try { return localStorage.getItem(storageKey); } catch (_) { return null; } }
  function sessionStatus(session) { return session?.analysis?.status || session?.status || "queued"; }
  function familyText(value) {
    const text = String(value || "待识别");
    return ["AIA", "HMI", "CME", "EPB", "DMSP", "ULF"].find((family) => text.toUpperCase().includes(family)) || text;
  }
  function datasetName(session) { return session?.dataset?.name || session?.dataset?.id || "未命名数据集"; }
  function updateSessionList(session) {
    if (!session?.id) return;
    const index = state.sessions.findIndex((item) => item.id === session.id);
    if (index < 0) state.sessions.unshift(session); else state.sessions[index] = { ...state.sessions[index], ...session };
    renderSessions();
  }
  function renderSessions() {
    const list = $("session-list");
    list.replaceChildren();
    $("session-count").textContent = state.sessions.length ? String(state.sessions.length) : "";
    if (!state.sessions.length) { list.append(element("p", "sidebar-empty", "还没有数据集对话。添加一个数据包，开始第一次检查。")); return; }
    for (const session of state.sessions) {
      const button = element("button", `session-item${session.id === state.activeId ? " active" : ""}`);
      button.type = "button";
      button.title = datasetName(session);
      button.setAttribute("aria-current", session.id === state.activeId ? "page" : "false");
      button.append(element("span", "session-name", datasetName(session)));
      const meta = element("span", "session-meta");
      const status = sessionStatus(session);
      meta.append(element("span", `status-dot ${safeStatus(status)}`), element("span", "", familyText(session.dataset?.family)), element("span", "", "·"), element("span", "", LABELS[status] || "状态待更新"));
      button.append(meta);
      button.addEventListener("click", () => selectSession(session.id));
      list.append(button);
    }
  }
  function safeStatus(value) { return /^[a-z_]+$/.test(String(value)) ? String(value) : "pending"; }
  async function loadSessions(selectInitial = false) {
    const token = ++state.listToken;
    $("refresh-sessions").disabled = true;
    try {
      const result = await request("/api/chat/sessions");
      if (token !== state.listToken) return;
      state.sessions = Array.isArray(result?.items) ? result.items : [];
      renderSessions();
      if (selectInitial && !state.activeId && new URLSearchParams(location.search).get("new") !== "1") {
        const saved = remembered();
        const id = state.sessions.some((item) => item.id === saved) ? saved : state.sessions[0]?.id;
        if (id) await selectSession(id);
      }
    } catch (error) {
      if (token !== state.listToken) return;
      $("session-list").replaceChildren(element("p", "sidebar-empty", "暂时无法读取会话。已有数据不会因刷新页面而删除。"));
      showNotice(errorMessage(error), () => loadSessions(selectInitial));
    } finally { if (token === state.listToken) $("refresh-sessions").disabled = false; }
  }
  function closeSidebar() {
    const wasOpen = $("sidebar").classList.contains("open");
    $("sidebar").classList.remove("open"); $("sidebar-scrim").hidden = true; $("mobile-menu").setAttribute("aria-expanded", "false");
    document.querySelector(".main-panel").inert = false;
    $("sidebar").removeAttribute("role"); $("sidebar").removeAttribute("aria-modal");
    if (wasOpen) $("mobile-menu").focus();
  }
  function openSidebar() {
    $("sidebar").classList.add("open"); $("sidebar-scrim").hidden = false; $("mobile-menu").setAttribute("aria-expanded", "true");
    $("sidebar").setAttribute("role", "dialog"); $("sidebar").setAttribute("aria-modal", "true");
    document.querySelector(".main-panel").inert = true;
    $("new-session").focus();
  }
  function showWelcome() {
    stopPoll(); state.readController?.abort(); state.viewToken++;
    resetWorkspace();
    state.activeId = null; state.current = null; state.renderedKey = "";
    remember(null); closeSidebar(); hideNotice(); renderSessions();
    $("welcome").hidden = false; $("conversation-content").hidden = true;
    $("conversation-title").textContent = "数据集质量助手";
    $("conversation-subtitle").textContent = "让每一个质量结论都有据可查";
    $("model-badge").classList.remove("available"); $("model-label").textContent = "分析工作空间";
    $("model-badge").title = "选择数据集后显示回答模型的实际连接状态";
    $("message-input").value = ""; resizeInput();
    $("conversation-scroll").scrollTop = 0;
    const url = new URL(location.href); url.searchParams.set("new", "1"); history.replaceState(null, "", url.pathname + url.search + url.hash);
  }
  function stopPoll() { clearTimeout(state.pollTimer); state.pollTimer = null; }
  async function selectSession(id, supplied) {
    if (!id) return;
    stopPoll(); state.readController?.abort();
    const token = ++state.viewToken;
    state.activeId = id; state.current = null; state.renderedKey = "";
    resetWorkspace();
    const url = new URL(location.href); url.searchParams.delete("new"); history.replaceState(null, "", url.pathname + url.search + url.hash);
    remember(id); closeSidebar(); hideNotice(); renderSessions();
    $("welcome").hidden = true; $("conversation-content").hidden = false;
    $("analysis-panel").hidden = true; $("messages").replaceChildren();
    $("conversation-placeholder").textContent = "正在读取会话…"; $("conversation-placeholder").hidden = false;
    $("conversation-title").textContent = "正在打开数据集…";
    updateComposer();
    if (supplied) { applySession(supplied); schedulePoll(); return; }
    await fetchActive(token);
  }
  async function fetchActive(token = state.viewToken) {
    const id = state.activeId;
    if (!id) return;
    state.readController?.abort();
    const controller = new AbortController(); state.readController = controller;
    const revision = state.revision.get(id) || 0;
    try {
      const session = await request(`/api/chat/sessions/${encodeURIComponent(id)}`, { signal: controller.signal });
      if (token !== state.viewToken || state.activeId !== id || revision !== (state.revision.get(id) || 0)) return;
      if (!session?.id || session.id !== id) throw new Error("会话返回编号不一致，请刷新后重试。");
      if ($("notice").classList.contains("error")) hideNotice();
      applySession(session); schedulePoll();
    } catch (error) {
      if (controller.signal.aborted || token !== state.viewToken || state.activeId !== id) return;
      $("conversation-placeholder").hidden = Boolean(state.current);
      if (!state.current) $("conversation-placeholder").textContent = "无法读取这个会话，请重试。";
      showNotice(errorMessage(error), () => { hideNotice(); fetchActive(); }, "重新连接");
      schedulePoll();
    }
  }
  function applySession(session, mutation = false) {
    updateSessionList(session);
    if (mutation) state.revision.set(session.id, (state.revision.get(session.id) || 0) + 1);
    if (session.id !== state.activeId) return;
    state.current = session;
    $("welcome").hidden = true; $("conversation-content").hidden = false; $("conversation-placeholder").hidden = true;
    $("conversation-title").textContent = datasetName(session);
    $("conversation-subtitle").textContent = `${familyText(session.dataset?.family)} · ${session.analysis?.source === "saved_report" ? "已载入历史报告" : LABELS[sessionStatus(session)] || "状态待更新"} · 每条结论可继续追问`;
    const llm = session.llm || {};
    $("model-badge").classList.toggle("available", Boolean(llm.available));
    $("model-label").textContent = llm.available ? (llm.model || "语言模型已接入") : "检查结果与证据问答";
    $("model-badge").title = llm.available ? `回答模型：${llm.model || "已连接"}` : "未接入语言模型时，展示实际检查结果和已有解释。";
    renderAnalysis(session); renderMessages(session); renderLiveRun(session); updateComposer();
    ensureInsights(session);
  }
  function schedulePoll() {
    stopPoll();
    if (!state.current || (TERMINAL.has(sessionStatus(state.current)) && !["queued", "running"].includes(state.current.liveRun?.status) && !(state.current.messages || []).some((message) => message.pending))) return;
    const token = state.viewToken;
    state.pollTimer = setTimeout(() => fetchActive(token), 2000);
  }
  function secondsText(seconds) {
    if (!Number.isFinite(Number(seconds)) || seconds === null) return "";
    const amount = Math.max(0, Math.floor(Number(seconds)));
    if (amount < 60) return `已用 ${amount} 秒`;
    const minutes = Math.floor(amount / 60), rest = amount % 60;
    return minutes < 60 ? `已用 ${minutes} 分 ${rest} 秒` : `已用 ${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分`;
  }
  function renderLiveRun(session) {
    const run = session.liveRun || {}, busy = ["queued", "running"].includes(run.status);
    const supported = ["EPB_seg_sample", "AIA_AR_sample"].includes(session.dataset?.family);
    $("live-run-button").disabled = busy || !supported;
    $("live-run-button").textContent = busy ? "正在重评…" : "现场重评整包 ▶";
    const panel = $("live-run-status"); panel.hidden = !run.status; panel.replaceChildren();
    if (!run.status) return;
    const header = element("div", "live-progress-heading");
    header.append(element("strong", "", run.phase || LABELS[run.status]), element("span", "", secondsText(run.elapsedSeconds)));
    const track = element("div", "progress-track"), fill = element("span");
    fill.style.width = Math.max(0, Math.min(100, run.percent || 0)) + "%"; track.append(fill);
    panel.append(header, track, element("small", "", "运行编号 " + run.runId + (run.forwardedImages !== undefined ? ` · 本次图像前向 ${run.forwardedImages} 张` : "")));
    if (run.error) panel.append(element("p", "live-error", run.error));
    if (run.status === "completed") panel.append(element("p", "live-completed", `已完成 ${run.audit?.records ?? ""} 条记录的整包重评；${run.audit?.freshRgbForwards ?? ""} 张图像重新前向，8类融合输出 ${run.audit?.freshFusionSlots ?? ""} 项。下方已切换本次新结果。`));
  }
  async function startLiveRun() {
    const id = state.activeId; if (!id) return;
    $("live-run-button").disabled = true;
    try {
      const session = await request(`/api/chat/sessions/${encodeURIComponent(id)}/live-run`, { method: "POST", body: JSON.stringify({}) });
      applySession(session, true); schedulePoll();
    } catch (error) { showNotice(errorMessage(error)); renderLiveRun(state.current); }
  }
  function renderAnalysis(session) {
    const analysis = session.analysis;
    $("analysis-panel").hidden = !analysis;
    if (!analysis) return;
    const status = sessionStatus(session), saved = analysis.source === "saved_report", running = !saved && !TERMINAL.has(status);
    $("analysis-title").textContent = saved ? "已载入历史评价报告" : LABELS[status] || "检查状态待更新";
    $("analysis-phase").textContent = analysis.phase || (running ? "等待服务器返回检查进展…" : status === "partial" ? "部分检查未完成，请结合步骤和报告查看覆盖范围。" : status === "failed" ? "本次检查未完成，可查看失败步骤后重试。" : "结果已保留，可以继续追问细节。" );
    $("analysis-elapsed").textContent = secondsText(analysis.elapsedSeconds);
    const evaluatedAt = analysis.evaluatedAt ? readableDate(analysis.evaluatedAt) : "评价时间未提供";
    $("analysis-source").hidden = !saved && analysis.source !== "live";
    $("analysis-source").textContent = saved ? `历史报告 · ${analysis.reportLabel || "已有分析结果"} · ${evaluatedAt}。本次载入未重新执行模型推理。` : "正在执行本次分析；进度与结果来自服务器实际运行。";
    $("analysis-panel").classList.toggle("saved-report", saved);
    $("analysis-icon").textContent = running ? "◌" : status === "failed" ? "!" : status === "partial" ? "◐" : "✓";
    $("analysis-icon").classList.toggle("spinning", running);
    const progress = analysis.progress || {}, processed = Number(progress.processed), total = Number(progress.total);
    const known = Number.isFinite(processed) && Number.isFinite(total) && total > 0 && progress.processed !== null;
    const fraction = known ? Math.max(0, Math.min(1, processed / total)) : 0;
    $("progress-count").textContent = known ? `已处理 ${processed.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")} 条` : running ? "正在获取处理数量" : "未提供处理数量";
    $("progress-percent").textContent = known ? `${(fraction * 100).toFixed(0)}%` : "";
    $("analysis-progress").classList.toggle("indeterminate", !known && running);
    $("analysis-progress-fill").style.width = `${fraction * 100}%`;
    if (known) { $("analysis-progress").setAttribute("aria-valuenow", String(processed)); $("analysis-progress").setAttribute("aria-valuemax", String(total)); $("analysis-progress").setAttribute("aria-valuemin", "0"); }
    else { $("analysis-progress").removeAttribute("aria-valuenow"); $("analysis-progress").removeAttribute("aria-valuemax"); }
    const steps = Array.isArray(analysis.steps) ? analysis.steps : [];
    $("tool-details").hidden = !steps.length;
    $("steps-count").textContent = `${steps.filter((step) => ["completed", "succeeded", "pass"].includes(step.status)).length} / ${steps.length} 已完成`;
    $("analysis-steps").replaceChildren();
    for (const step of steps) { const item = element("li", `tool-step ${safeStatus(step.status)}`); item.append(element("span", "", step.label || "检查步骤"), element("span", "", LABELS[step.status] || step.status || "待执行")); $("analysis-steps").append(item); }
    $("retry-analysis").hidden = !["failed", "partial"].includes(status);
    $("retry-analysis").disabled = state.creating;
  }
  function safeUrl(value) {
    if (typeof value !== "string" || !value.trim()) return null;
    try { const url = new URL(value, location.origin); return ["http:", "https:"].includes(url.protocol) ? url.href : null; } catch (_) { return null; }
  }
  function messageNode(message) {
    const role = message.role === "user" ? "user" : "assistant";
    const article = element("article", `message ${role}${message.pending ? " pending" : ""}`);
    if (role === "assistant") { const header = element("div", "assistant-header"); header.append(element("span", "assistant-avatar", "✳"), element("span", "", "SpaceQA")); article.append(header); }
    if (role === "assistant" && Array.isArray(message.cards) && message.cards.length) {
      const cards = element("div", "metric-cards");
      for (const card of message.cards) { const node = element("div", "metric-card"); node.append(element("strong", "", card.value === null || card.value === undefined ? "待统计" : card.value), element("span", "", card.label || "检查指标")); cards.append(node); }
      article.append(cards);
    }
    const content = element("div", "message-content", message.content || (message.pending ? "正在组织回答…" : ""));
    article.append(content);
    if (message.pending) { const dots = element("span", "thinking-dots"); dots.setAttribute("aria-label", "正在处理"); for (let i = 0; i < 3; i++) dots.append(element("i")); content.append(dots); }
    if (role === "assistant" && Array.isArray(message.evidence) && message.evidence.length) {
      const links = element("div", "evidence-links");
      for (const evidence of message.evidence) { const url = safeUrl(evidence.url); if (!url) continue; const link = element("a", "evidence-link", `${evidence.label || "查看样本证据"} ↗`); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; links.append(link); }
      if (links.childElementCount) { article.append(links, element("p", "evidence-caption", "查看证据时，请区分原始图像、输入标注、模型候选与已确认缺陷。")); }
    }
    return article;
  }
  function renderMessages(session, force = false) {
    const messages = Array.isArray(session.messages) ? session.messages : [];
    const sending = state.sending.get(session.id);
    const key = JSON.stringify([session.id, messages, sending?.startedAt]);
    if (!force && key === state.renderedKey) return;
    state.renderedKey = key;
    const scroll = $("conversation-scroll"), nearBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 130;
    const list = $("messages"); list.replaceChildren();
    for (const message of messages) list.append(messageNode(message));
    if (sending) {
      const serverHasUser = messages.slice(sending.baseCount).some((message) => message.role === "user" && message.content === sending.content);
      if (!serverHasUser) list.append(messageNode({ role: "user", content: sending.content }));
      if (!messages.some((message) => message.pending && message.role !== "user")) list.append(messageNode({ role: "assistant", content: "正在结合检查结果组织回答", pending: true }));
    }
    if (!messages.length && !sending) list.append(messageNode({ role: "assistant", content: TERMINAL.has(sessionStatus(session)) ? "会话已打开，服务器暂未提供可显示的总结。你可以追问已有检查结果或查看报告证据。" : "数据集已接入，正在等待服务器返回检查结果。完成后会在这里展示总体结论和可追溯证据。", pending: !TERMINAL.has(sessionStatus(session)) }));
    if (force || nearBottom) requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
  }
  function updateComposer() {
    const hasDataset = Boolean(state.current?.id), sending = state.sending.has(state.activeId);
    $("message-input").disabled = !hasDataset;
    $("message-input").placeholder = hasDataset ? "追问这个数据集的问题、位置或判断依据…" : "选择数据集后，追问它的质量、问题与证据…";
    $("send-message").disabled = !hasDataset || sending || !$("message-input").value.trim();
    $("composer-hint").textContent = sending ? "正在组织回答，请稍候…" : hasDataset ? "Enter 发送 · Shift + Enter 换行" : "每个数据集保留一段独立对话";
  }
  function resizeInput() { const input = $("message-input"); input.style.height = "auto"; input.style.height = `${Math.min(170, input.scrollHeight)}px`; updateComposer(); }
  async function sendMessage(event) {
    event?.preventDefault();
    const id = state.activeId, current = state.current, content = $("message-input").value.trim();
    if (!id || !current || !content || state.sending.has(id)) return;
    setWorkspace("conversation");
    state.sending.set(id, { content, startedAt: Date.now(), baseCount: (current.messages || []).length });
    $("message-input").value = ""; resizeInput(); hideNotice(); renderMessages(current, true);
    try {
      const session = await post(`/api/chat/sessions/${encodeURIComponent(id)}/messages`, { message: content }, 150000);
      if (!session?.id) throw new Error("回答已返回，但会话格式不完整。请刷新查看。");
      state.sending.delete(id); applySession(session, true);
      if (state.activeId === id) { renderMessages(session, true); schedulePoll(); }
    } catch (error) {
      state.sending.delete(id);
      if (state.activeId !== id) return;
      if (!$("message-input").value) $("message-input").value = content;
      resizeInput(); renderMessages(state.current || current, true);
      showNotice(errorMessage(error), () => { hideNotice(); fetchActive(); }, "刷新会话");
    } finally { if (state.activeId === id) updateComposer(); }
  }
  async function retryAnalysis() {
    const id = state.activeId;
    if (!id || state.creating) return;
    state.creating = true; $("retry-analysis").disabled = true; hideNotice();
    try { const session = await post(`/api/chat/sessions/${encodeURIComponent(id)}/retry`, {}, 60000); applySession(session, true); if (state.activeId === id) schedulePoll(); }
    catch (error) { if (state.activeId === id) showNotice(errorMessage(error), retryAnalysis, "重试分析"); }
    finally { state.creating = false; $("retry-analysis").disabled = false; }
  }
  function dialogError(message) { $("dialog-error").textContent = message || ""; $("dialog-error").hidden = !message; }
  function setTab(name) {
    const existing = name !== "upload";
    $("tab-existing").classList.toggle("active", existing); $("tab-existing").setAttribute("aria-selected", String(existing));
    $("tab-upload").classList.toggle("active", !existing); $("tab-upload").setAttribute("aria-selected", String(!existing));
    $("panel-existing").hidden = !existing; $("panel-upload").hidden = existing;
    if (existing && !state.folderLoaded) browseFolder();
  }
  function openDatasetDialog(tab = "existing") { dialogError(""); if (!$("dataset-dialog").open) $("dataset-dialog").showModal(); setTab(tab); }
  function closeDatasetDialog() { if (!state.upload) $("dataset-dialog").close(); }
  function readableDate(value) { const parsed = new Date(value); return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString("zh-CN", { hour12: false }); }
  async function browseFolder(folderPath = "") {
    const token = ++state.folderToken;
    state.folderLoading = true;
    state.selectedDataset = null; renderFolderSelection();
    $("dataset-list").replaceChildren(element("p", "dialog-empty", "正在读取服务器目录…"));
    try {
      const result = await request(`/api/chat/folders${folderPath ? `?path=${encodeURIComponent(folderPath)}` : ""}`);
      if (token !== state.folderToken) return;
      state.folders = Array.isArray(result?.folders) ? result.folders : [];
      state.folderPath = result.currentPath || ""; state.folderLoaded = true;
      state.selectedDataset = result.canSelect !== false && result.selectedDataset ? result.selectedDataset : null;
      $("dataset-search").value = "";
      $("folder-breadcrumbs").replaceChildren();
      const root = element("button", "", "可访问目录"); root.type = "button"; root.addEventListener("click", () => browseFolder()); $("folder-breadcrumbs").append(root);
      for (const crumb of Array.isArray(result.breadcrumbs) ? result.breadcrumbs : []) { $("folder-breadcrumbs").append(element("span", "", "/")); const button = element("button", "", crumb.name || crumb.path); button.type = "button"; button.title = crumb.path || ""; button.addEventListener("click", () => browseFolder(crumb.path)); $("folder-breadcrumbs").append(button); }
      $("family-filter").replaceChildren(new Option("全部类型", ""));
      [...new Set(state.folders.filter((item) => item.isDataset || item.datasetId).map((item) => familyText(item.family || item.name)))].sort().forEach((family) => $("family-filter").append(new Option(family, family)));
      renderFolders(); renderFolderSelection();
    } catch (error) {
      if (token !== state.folderToken) return;
      const retry = element("button", "retry-datasets", "重新读取目录"); retry.type = "button"; retry.addEventListener("click", () => browseFolder(folderPath));
      $("dataset-list").replaceChildren(element("p", "dialog-empty", errorMessage(error)), retry);
    } finally { if (token === state.folderToken) state.folderLoading = false; }
  }
  function renderFolders() {
    const query = $("dataset-search").value.trim().toLocaleLowerCase(), family = $("family-filter").value;
    const matches = state.folders.filter((item) => (!query || `${item.name || ""} ${item.path || ""}`.toLocaleLowerCase().includes(query)) && (!family || familyText(item.family || item.name) === family));
    $("dataset-list").replaceChildren();
    if (!matches.length) { $("dataset-list").append(element("p", "dialog-empty", state.folders.length ? "没有符合条件的文件夹，试试其他名称或类型。" : state.selectedDataset ? "已识别当前数据集，点击下方按钮查看分析。" : "此目录没有可浏览的文件夹。")); return; }
    for (const folder of matches) {
      const isDataset = Boolean(folder.isDataset || folder.datasetId);
      const row = element("div", `folder-row${state.selectedDataset?.id === folder.datasetId && folder.datasetId ? " selected" : ""}`);
      const item = element("button", "folder-main"); item.type = "button"; item.disabled = state.creating;
      item.append(element("span", `folder-icon${isDataset ? " dataset" : ""}`, "▱"));
      const info = element("span", "dataset-info"); info.append(element("strong", "", folder.name || folder.path || "文件夹"), element("small", "", isDataset ? "可添加的数据集" : "打开文件夹"));
      item.append(info, element("span", "folder-type", isDataset ? familyText(folder.family || folder.name) : "文件夹"));
      if (isDataset) item.setAttribute("aria-pressed", String(Boolean(folder.datasetId && state.selectedDataset?.id === folder.datasetId)));
      item.setAttribute("aria-label", `${folder.datasetId ? "选择数据集" : "打开文件夹"}：${folder.name || folder.path || "未命名"}`);
      item.addEventListener("click", () => { if (folder.datasetId) { state.selectedDataset = { id: folder.datasetId, name: folder.name, family: folder.family || familyText(folder.name) }; renderFolders(); renderFolderSelection(); } else browseFolder(folder.path); });
      const open = element("button", "folder-open", "›"); open.type = "button"; open.title = `打开 ${folder.name || "文件夹"}`; open.setAttribute("aria-label", open.title); open.addEventListener("click", () => browseFolder(folder.path));
      row.append(item, open); $("dataset-list").append(row);
    }
  }
  function renderFolderSelection() {
    $("selected-folder-label").textContent = state.selectedDataset ? state.selectedDataset.name || state.selectedDataset.id : "选择一个数据集文件夹";
    $("selected-folder-description").textContent = state.creating ? "正在读取已有评价结果…" : state.selectedDataset ? `${familyText(state.selectedDataset.family)} · 优先展示已有报告` : "浏览服务器目录，找到想要查看的数据包";
    $("add-selected-dataset").disabled = !state.selectedDataset || state.creating;
    $("add-selected-dataset").textContent = state.creating ? "正在打开…" : "添加并查看分析";
  }
  async function createSession(dataset) {
    if (!dataset?.id || state.creating || state.upload) return;
    state.creating = true; dialogError(""); renderFolders(); renderFolderSelection();
    $("demo-dataset").disabled = true;
    if (!$("dataset-dialog").open) showNotice("正在读取数据集已有评价结果…", null, "", false);
    try {
      const session = await post("/api/chat/sessions", { datasetId: dataset.id, preferCached: true }, 60000);
      if (!session?.id) throw new Error("数据集已提交，但缺少会话编号。请刷新会话列表确认。");
      $("dataset-dialog").close(); await selectSession(session.id, session);
    } catch (error) { if ($("dataset-dialog").open) dialogError(errorMessage(error)); else showNotice(errorMessage(error), () => createSession(dataset)); }
    finally { state.creating = false; $("demo-dataset").disabled = false; renderFolders(); renderFolderSelection(); }
  }
  async function loadDemo() {
    try { const demo = await request("/api/chat/demo"); if (!demo.available || !demo.dataset?.id) return; state.demo = demo; $("demo-title").textContent = demo.title || "示例数据集"; $("demo-description").textContent = demo.description || "拖入对话区，查看已有质量报告并继续追问。"; $("demo-name").textContent = demo.dataset.name || demo.dataset.id; $("demo-dataset").hidden = false; }
    catch (_) { /* A sample is optional. Normal dataset access stays available. */ }
  }
  function sizeText(bytes) { return bytes < 1024 ** 2 ? `${(bytes / 1024).toFixed(1)} KB` : bytes < 1024 ** 3 ? `${(bytes / 1024 ** 2).toFixed(1)} MB` : `${(bytes / 1024 ** 3).toFixed(2)} GB`; }
  function uploadZip(file) {
    if (!file || state.upload || state.creating) return;
    if (!/\.zip$/i.test(file.name)) { dialogError("请选择 ZIP 压缩包。文件夹或其他压缩格式请先在服务器整理，再选取已有数据包。"); return; }
    if (!file.size) { dialogError("这个文件为空，请选择包含数据与标注的 ZIP。"); return; }
    dialogError(""); setTab("upload");
    $("upload-name").textContent = file.name; $("upload-name").title = file.name;
    $("upload-status").textContent = `准备上传 · ${sizeText(file.size)}`;
    $("upload-progress").hidden = false; $("drop-zone").hidden = true;
    $("upload-progress-fill").style.width = "0%"; $("upload-progress-track").classList.remove("indeterminate");
    $("upload-progress-track").removeAttribute("aria-valuenow");
    const xhr = new XMLHttpRequest(); state.upload = xhr;
    $("dialog-close").disabled = true; $("tab-existing").disabled = true;
    $("cancel-upload").disabled = false;
    xhr.open("POST", "/api/chat/uploads");
    xhr.responseType = "text";
    xhr.setRequestHeader("Content-Type", "application/zip"); xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name)); xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.min(100, (event.loaded / event.total) * 100);
      $("upload-status").textContent = `已上传 ${sizeText(event.loaded)} / ${sizeText(event.total)} · ${percent.toFixed(0)}%`;
      $("upload-progress-fill").style.width = `${percent}%`;
      $("upload-progress-track").setAttribute("aria-valuenow", String(event.loaded)); $("upload-progress-track").setAttribute("aria-valuemax", String(event.total));
    };
    xhr.upload.onload = () => { $("upload-status").textContent = "传输完成，服务器正在接收并建立数据集…"; $("upload-progress-track").classList.add("indeterminate"); $("cancel-upload").disabled = true; };
    const finish = () => { state.upload = null; $("dialog-close").disabled = false; $("tab-existing").disabled = false; $("zip-input").value = ""; $("upload-progress-track").classList.remove("indeterminate"); $("drop-zone").hidden = false; $("upload-progress").hidden = true; };
    xhr.onload = async () => {
      let session;
      try { session = JSON.parse(xhr.responseText); } catch (_) { finish(); dialogError(`服务器未返回可读取的会话（HTTP ${xhr.status}）。`); return; }
      if (xhr.status < 200 || xhr.status >= 300 || !session?.id) { finish(); dialogError(session?.error?.message || session?.message || (typeof session?.error === "string" ? session.error : `上传未完成（HTTP ${xhr.status}）。`)); return; }
      finish(); connection(true); state.folderLoaded = false; $("dataset-dialog").close(); await selectSession(session.id, session);
    };
    xhr.onerror = () => { finish(); connection(false); dialogError("上传连接中断。请检查服务与连接后重新选择 ZIP；大型数据包建议从服务器选取。"); };
    xhr.onabort = () => { finish(); dialogError("上传已取消。"); };
    xhr.send(file);
  }

  function finite(value) { return typeof value === "number" && Number.isFinite(value); }
  function displayNumber(value, suffix = "") { return finite(value) ? value.toLocaleString("zh-CN", { maximumFractionDigits: 1 }) + suffix : "—"; }
  function textValue(value) { if (value === null || value === undefined) return "未提供"; if (typeof value === "string") return value; return JSON.stringify(value, null, 2); }
  function stageName(value) { return value === "confirmed" ? "已确认问题" : value === "clean" ? "干净样本" : "待复核候选"; }
  function defectName(code) { return DEFECTS.find((item) => item[0] === code)?.[1] || code || "未分类"; }
  function axisState(axis) {
    if (typeof axis.stateLabel === "string" && axis.stateLabel.trim()) return axis.stateLabel.replace(/待判断|待判/g, "未覆盖");
    const status = axis.displayState || axis.executionState || axis.execution?.state || axis.state;
    return ({ completed: "已执行", succeeded: "已执行", full: "已执行", partial: "部分覆盖", not_applicable: "不适用", undetermined: "未覆盖", not_executed: "未执行", not_run: "未执行", uncovered: "未覆盖", skipped: "未执行", running: "执行中", queued: "未执行", failed: "执行失败" })[status] || (axis.applicable === false ? "不适用" : "未覆盖");
  }
  function axisDisplayScore(axis) {
    const valid = (value) => finite(value) && value >= 0 && value <= 100;
    if (Object.prototype.hasOwnProperty.call(axis, "displayScore")) return valid(axis.displayScore) ? axis.displayScore : null;
    if (axis.applicable === false || axis.state === "not_applicable") return null;
    if (valid(axis.candidateProfileScore)) return axis.candidateProfileScore;
    return valid(axis.qualityScore) ? axis.qualityScore : null;
  }
  function caseLocationLabel(item) {
    const locator = item.locator || {};
    if (Array.isArray(locator.bbox) && locator.bbox.length === 4 && locator.bbox.every(finite) && locator.bbox[2] > locator.bbox[0] && locator.bbox[3] > locator.bbox[1]) return "含区域定位";
    const intervals = [locator.timeInterval, locator.time_interval, locator.interval, locator.time_range];
    if (intervals.some((value) => Array.isArray(value) && value.length === 2 && value.every((endpoint) => typeof endpoint === "string" ? Boolean(endpoint.trim()) : finite(endpoint)))) return "含时段定位";
    if (locator.annotation_pointer || locator.annotationPointer) return "含字段定位";
    if (locator.type === "file" || locator.expected_data_path) return "含文件定位";
    if (locator.type === "annotationRecord") return "含标注记录";
    return "样本级定位";
  }
  function localUrl(value) { const url = safeUrl(value); return url && new URL(url).origin === location.origin ? url : null; }
  function resetWorkspace() {
    workspace.insightController?.abort(); workspace.casesController?.abort(); workspace.tierController?.abort(); workspace.detailController?.abort(); workspace.explainController?.abort();
    workspace.casesToken++; workspace.tierToken++; workspace.detailToken++; workspace.key = ""; workspace.insights = null; workspace.detail = null; workspace.offset = 0; workspace.total = 0; workspace.tier = "";
    clearTimeout(workspace.searchTimer);
    if ($("case-dialog").open) $("case-dialog").close();
    $("insights-loading").hidden = false; $("insights-loading").textContent = "正在读取八类缺陷统计…"; $("insights-content").hidden = true;
    $("case-defect-filter").value = ""; $("case-stage-filter").value = ""; $("case-search").value = ""; $("case-count-badge").textContent = "";
    $("case-list").replaceChildren(); setWorkspace("overview", false);
    $("tier-sample-panel").hidden = true;
  }
  function setWorkspace(view, load = true) {
    workspace.view = view;
    for (const name of ["overview", "cases", "conversation"]) { const selected = name === view; $("view-" + name).classList.toggle("active", selected); $("view-" + name).setAttribute("aria-pressed", String(selected)); $(name === "overview" ? "insights-view" : name + "-view").hidden = !selected; }
    if (load && view === "cases" && state.current) loadCases();
    if (load) $("conversation-scroll").scrollTop = 0;
  }
  function ensureInsights(session, force = false) {
    const analysis = session.analysis || {};
    $("workspace-source").textContent = analysis.source === "saved_report" ? `历史评价 · ${analysis.reportLabel || "已保存报告"} · ${analysis.evaluatedAt ? readableDate(analysis.evaluatedAt) : "时间未提供"} · 本次载入未重新推理` : analysis.source === "live_recompute" ? `本次真实重评 · ${analysis.evaluatedAt ? readableDate(analysis.evaluatedAt) : ""} · 文件及规则、八类融合、全部可读图像定位已重新执行；上游专家预测缓存复用` : "当前运行的检查结果；检查完成后自动更新画像与案例。";
    const key = [session.id, analysis.taskId, analysis.evaluatedAt, sessionStatus(session)].join(":");
    if (!force && key === workspace.key) return;
    workspace.key = key;
    if (!analysis.result && !TERMINAL.has(sessionStatus(session))) { $("insights-loading").textContent = "数据集正在分析。点击“对话追问”可查看实际检查进度，完成后显示八类画像。"; if (!analysis.result) setWorkspace("conversation", false); return; }
    loadInsights(session.id);
  }
  async function loadInsights(id = state.activeId) {
    if (!id) return;
    workspace.insightController?.abort(); const controller = new AbortController(); workspace.insightController = controller;
    $("insights-loading").hidden = false; $("insights-loading").textContent = "正在读取八类缺陷统计…";
    try {
      const value = await request(`/api/chat/sessions/${encodeURIComponent(id)}/insights`, { signal: controller.signal, timeout: 45000 });
      if (controller.signal.aborted || state.activeId !== id) return;
      workspace.insights = value; renderInsights(value); $("insights-loading").hidden = true; $("insights-content").hidden = false;
    } catch (error) {
      if (controller.signal.aborted || state.activeId !== id) return;
      const retry = element("button", "text-button", "重新读取 →"); retry.type = "button"; retry.addEventListener("click", () => loadInsights(id));
      $("insights-loading").replaceChildren(element("p", "", errorMessage(error)), retry);
    }
  }
  function renderInsights(value) {
    const summary = value.summary || {}, facts = state.current?.analysis?.result || {};
    $("insight-metrics").replaceChildren();
    for (const [label, amount] of [["登记样本", summary.totalSamples ?? facts.totalSamples], ["确认问题样本", summary.confirmedSamples ?? facts.confirmedSamples], ["仅待复核样本", summary.candidateOnlySamples ?? facts.candidateOnlySamples], ["去重检出比例", finite(summary.detectedRatio ?? facts.detectedRatio) ? displayNumber((summary.detectedRatio ?? facts.detectedRatio) * 100, "%") : "—"]]) { const card = element("div", "insight-metric"); card.append(element("strong", "", finite(amount) ? displayNumber(amount) : amount ?? "—"), element("span", "", label)); $("insight-metrics").append(card); }
    renderPollution(value.pollution || {});
    renderTierCards(Array.isArray(value.tiers) ? value.tiers : []);
    const axes = DEFECTS.map(([code, name]) => ({ code, name, ...(Array.isArray(value.axes) ? value.axes.find((a) => a.code === code) : null) }));
    drawRadar(axes);
    $("radar-formula").textContent = "检出画像分 = 100 × (1 − 该类检出问题样本数 ÷ 登记样本数)，统计规则与模型实际检出的已确认问题和候选。候选保留待复核状态；未执行、未覆盖或不适用的维度不补分。八类画像与官方六维评价分别展示。";
    const inference = value.inferenceCoverage || {};
    if (finite(inference.actualFreshImageForwards) && finite(inference.expectedReadableImageSamples) && inference.expectedReadableImageSamples > 0) $("radar-formula").textContent += ` 本轮图像前向：${displayNumber(inference.actualFreshImageForwards)} / ${displayNumber(inference.expectedReadableImageSamples)}张；这是图像处理覆盖，每类问题的定位覆盖另见下表。`;
    else if (inference.status === "not_pixel_task") $("radar-formula").textContent += " 本包为时间序列任务，不执行原生图像前向。";
    $("axis-table-body").replaceChildren();
    for (const axis of axes) {
      const row = element("tr", (axis.displayState || axis.state) === "not_applicable" ? "axis-na" : "");
      const name = element("td"); name.append(element("strong", "", axis.name), element("small", "", axis.code));
      const displayScore = axisDisplayScore(axis);
      const score = element("td"); score.append(element("strong", "axis-score", finite(displayScore) ? displayNumber(displayScore) : axisState(axis)));
      if (finite(axis.confirmedQualityScore) && finite(displayScore) && axis.confirmedQualityScore !== displayScore) score.append(element("small", "candidate-score", "仅确认参考 " + displayNumber(axis.confirmedQualityScore)));
      const status = element("td"); status.append(element("span", `axis-status ${safeStatus(axis.displayState || axis.state)}`, axisState(axis)));
      if (finite(axis.classificationCoverage?.ratio)) status.append(element("small", "", "分类执行 " + displayNumber(axis.classificationCoverage.ratio * 100, "%")));
      else if (finite(axis.coverage?.ratio)) status.append(element("small", "", "检查覆盖 " + displayNumber(axis.coverage.ratio * 100, "%")));
      const localization = axis.localizationCoverage || {};
      if (finite(localization.locatedSamples) && finite(localization.classifiedPositiveSamples)) status.append(element("small", "", `检出中定位 ${displayNumber(localization.locatedSamples)} / ${displayNumber(localization.classifiedPositiveSamples)}` + (localization.classifiedPositiveSamples > 0 ? `（${displayNumber(localization.locatedSamples / localization.classifiedPositiveSamples * 100, "%")}）` : "")));
      else if (finite(localization.ratio)) status.append(element("small", "", "定位覆盖 " + displayNumber(localization.ratio * 100, "%")));
      const action = element("td"), button = element("button", "text-button", "看案例 ↗"); button.type = "button"; button.setAttribute("aria-label", "查看" + axis.name + "案例"); button.addEventListener("click", () => showDefectCases(axis.code)); action.append(button);
      row.append(name, element("td", "count-confirmed", displayNumber(axis.confirmedSamples)), element("td", "count-candidate", displayNumber(axis.candidateOnlySamples)), score, status, action); $("axis-table-body").append(row);
    }
    renderSixDimensions(value.sixDimensions || value.coverage?.dimensions || {});
    const meaning = value.coverageMeaning || {};
    $("coverage-explanation").replaceChildren();
    for (const [label, key] of [["执行覆盖", "execution"], ["定位覆盖", "localization"], ["判定覆盖", "verdict"]]) {
      const item = element("div"); item.append(element("strong", "", label), element("p", "", meaning[key] || "详见逐项检查记录。")); $("coverage-explanation").append(item);
    }
    const formal = value.sixDimensionEvaluation?.formalEvaluation;
    if (formal) {
      const item = element("div", "coverage-formal-status");
      item.append(element("strong", "", formal.eligible ? "六维正式分：条件已满足" : "六维正式分：暂不能宣称全部达标"));
      const blockers = Array.isArray(formal.blockers) ? formal.blockers : [];
      item.append(element("p", "", blockers.length ? blockers.join("；") : "六维适用检查、规范材料和实现一致性均已记录。"));
      $("coverage-explanation").append(item);
    }
    $("coverage-gaps").replaceChildren();
    for (const gap of value.coverageGaps || []) { const item = element("div", "coverage-gap"); item.append(element("strong", "", `${gap.name} · ${gap.unresolvedVerdictSamples}条证据未决`), element("p", "", gap.reason), element("small", "", "下一步：" + gap.nextAction)); $("coverage-gaps").append(item); }
    if (!(value.coverageGaps || []).length) $("coverage-gaps").append(element("p", "", "本报告未记录内容检查的未决项；六维材料需求仍按维度展示。"));
    renderAuditPolicy(value);
    $("overview-export").replaceChildren();
    for (const [label, url] of [["打开完整质量报告 / 保存PDF ↗", `/api/chat/sessions/${encodeURIComponent(state.activeId)}/report.html`], ["下载分析摘要 JSON ↗", `/api/chat/sessions/${encodeURIComponent(state.activeId)}/report`], ["下载全部问题台账 JSONL ↗", `/api/chat/sessions/${encodeURIComponent(state.activeId)}/evidence.jsonl`], ["浏览全部数据与输入标注 ↗", "/index.html#datasets/" + encodeURIComponent(state.current?.dataset?.id || "")]]) { const link = element("a", "evidence-link", label); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer"; $("overview-export").append(link); }
  }

  function renderPollution(pollution) {
    const total = pollution.pollutionIndex;
    $("pollution-total").textContent = finite(total) ? "综合 " + displayNumber(total) : "当前无可复算概率";
    $("pollution-summary").replaceChildren();
    const summary = [
      ["统计分母", finite(pollution.sampleCount) ? displayNumber(pollution.sampleCount) + " 个登记样本" : "—"],
      ["排序方向", "污染指数升序（越低越好）"],
      ["阈值规则", "仅概率 ≥ 类别阈值计入"]
    ];
    for (const [label, value] of summary) { const card = element("div", "pollution-summary-card"); card.append(element("strong", "", value), element("span", "", label)); $("pollution-summary").append(card); }
    const defects = pollution.defects || {};
    $("pollution-table-body").replaceChildren();
    for (const [code, name] of DEFECTS) {
      const row = element("tr"), item = defects[code] || {};
      row.append(element("td", "", name + " · " + code), element("td", "", finite(item.threshold) ? displayNumber(item.threshold) : "—"), element("td", "", finite(item.overThresholdSamples) ? displayNumber(item.overThresholdSamples) : "—"), element("td", "", finite(item.probabilitySum) ? displayNumber(item.probabilitySum) : "—"), element("td", "", finite(item.pollutionIndexContribution) ? displayNumber(item.pollutionIndexContribution) : "—"), element("td", "", finite(item.missingProbabilitySamples) && item.missingProbabilitySamples > 0 ? "缺概率 " + displayNumber(item.missingProbabilitySamples) + " 条" : "已有类别概率或确定性证据"));
      $("pollution-table-body").append(row);
    }
    $("pollution-note").textContent = pollution.meaning || "污染指数来自当前规则/模型概率观测，没有独立真值时不能解释为官方污染率或召回率。";
  }

  function renderAuditPolicy(value) {
    const policy = value.localizationPolicy || {}, host = $("localization-policy"); host.replaceChildren();
    const tolerance = policy.tolerances || {};
    const entries = [["定位前提", policy.referenceRequired ? "必须有独立参考；没有时只保留候选" : "未声明"], ["框 IoU", tolerance.bbox?.iouThreshold ?? "—"], ["中心距离", tolerance.bbox?.normalizedCenterDistanceThreshold ?? "—"], ["多边形 IoU", tolerance.polygon?.iouThreshold ?? "—"], ["时间端点（秒）", tolerance.timeInterval?.endpointToleranceSeconds ?? "—"], ["尺寸相对误差", tolerance.metadata?.relativeDimensionError ?? "—"]];
    for (const [label, text] of entries) { const card = element("div", "coverage-policy-item"); card.append(element("strong", "", label), element("p", "", String(text))); host.append(card); }
    const audit = value.expertFusionAudit || {}, auditHost = $("expert-audit"); auditHost.replaceChildren();
    if (!audit.available) { auditHost.append(element("p", "", "本次报告没有保存专家原始概率，未虚构异常值剔除记录。启用融合审计后会在这里列出被剔除专家及原因。")); return; }
    auditHost.append(element("strong", "", "MAD 审计：" + displayNumber(audit.excludedCount || 0) + " 个输出被标记为异常"));
    for (const row of (audit.records || []).slice(0, 12)) { const item = element("div", "coverage-gap"); item.append(element("p", "", (row.sampleId || "样本") + " · " + (row.defectType || "融合") + " · 保留 " + displayNumber((row.keptExperts || []).length) + " 个专家"), element("small", "", row.excludedExperts?.length ? "剔除：" + row.excludedExperts.map((x) => x.name || x).join("、") : "没有超过MAD阈值的专家")); auditHost.append(item); }
  }

  function renderTierCards(tiers) {
    const host = $("tier-cards");
    host.replaceChildren();
    if (!tiers.length) { host.append(element("p", "panel-empty", "当前报告没有可展示的样本档案。")); return; }
    for (const tier of tiers) {
      const card = element("button", `tier-card ${tier.id === "clean" ? "clean" : "defect"}`); card.type = "button";
      const top = element("span", "tier-card-top"); top.append(element("span", "tier-card-code", tier.id === "clean" ? "P0 / CLEAN" : tier.id), element("span", "tier-card-arrow", "查看 →"));
      card.append(top, element("strong", "tier-card-name", tier.name), element("span", "tier-card-count", `${displayNumber(tier.sampleCount)} 个样本`));
      const meta = tier.id === "clean" ? "已完成检查且没有问题记录" : `已确认 ${displayNumber(tier.confirmedSampleCount)} · 候选 ${displayNumber(tier.candidateSampleCount)}`;
      card.append(element("small", "tier-card-meta", meta));
      if (tier.manyToMany) card.append(element("small", "tier-card-note", "允许与其他缺陷档案交叉"));
      card.addEventListener("click", () => showTier(tier.id));
      host.append(card);
    }
  }
  function svgNode(tag, attrs = {}, text) { const node = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value)); if (text !== undefined) node.textContent = text; return node; }
  function drawRadar(axes) {
    const svg = svgNode("svg", { viewBox: "0 0 580 460", role: "img", "aria-label": "八类检出画像分，越高表示检出问题越少，未覆盖的维度不补分" });
    svg.append(svgNode("title", {}, "八类检出画像分"));
    const center = [290, 221], radius = 145, point = (i, val, extra = 0) => { const angle = -Math.PI / 2 + i * Math.PI / 4; const r = radius * val / 100 + extra; return [center[0] + Math.cos(angle) * r, center[1] + Math.sin(angle) * r]; };
    for (const level of [25, 50, 75, 100]) { svg.append(svgNode("polygon", { points: axes.map((_, i) => point(i, level).join(",")).join(" "), fill: "none", stroke: "#dfe6d8", "stroke-width": 1 })); svg.append(svgNode("text", { x: center[0] + 6, y: center[1] - radius * level / 100 + 11, fill: "#a1ab97", "font-size": 9 }, String(level))); }
    axes.forEach((axis, i) => {
      const end = point(i, 100), label = point(i, 100, 43);
      svg.append(svgNode("line", { x1: center[0], y1: center[1], x2: end[0], y2: end[1], stroke: "#e0e7da", "stroke-dasharray": "3 4" }));
      const group = svgNode("g", { role: "button", tabindex: 0, "aria-label": "按" + axis.name + "筛选案例", class: "radar-axis-label" });
      const anchor = i === 0 || i === 4 ? "middle" : i < 4 ? "start" : "end";
      const shortNames = { "L0-01": "文件缺失", "L2-G": "边界偏移", "L2-P": "类别错误", "L2-T": "时间抖动", "L2-F": "尺寸矛盾", "L3-06": "漏标", "L3-07": "虚标", "L3-MIS": "覆盖偏差" };
      group.append(svgNode("text", { x: label[0], y: label[1] - 2, "text-anchor": anchor, fill: "#385c89", "font-size": 13 }, shortNames[axis.code] || axis.name));
      const displayScore = axisDisplayScore(axis);
      group.append(svgNode("text", { x: label[0], y: label[1] + 14, "text-anchor": anchor, fill: finite(displayScore) ? "#39786d" : "#9da793", "font-size": 10 }, finite(displayScore) ? displayNumber(displayScore) + "分" : axisState(axis)));
      group.addEventListener("click", () => showDefectCases(axis.code)); group.addEventListener("keydown", (e) => { if (["Enter", " "].includes(e.key)) { e.preventDefault(); showDefectCases(axis.code); } }); svg.append(group);
    });
    function drawProfile(values, color, dashed, fill) {
      if (values.every(finite)) svg.append(svgNode("polygon", { points: values.map((value, i) => point(i, value).join(",")).join(" "), fill, stroke: color, "stroke-width": 2, ...(dashed ? { "stroke-dasharray": "5 4" } : {}) }));
      else values.forEach((value, i) => { const next = (i + 1) % values.length; if (finite(value) && finite(values[next])) { const a = point(i, value), b = point(next, values[next]); svg.append(svgNode("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color, "stroke-width": 2, ...(dashed ? { "stroke-dasharray": "5 4" } : {}) })); } });
      values.forEach((value, i) => { if (!finite(value)) return; const p = point(i, value); const dot = svgNode("circle", { cx: p[0], cy: p[1], r: dashed ? 4 : 4.5, fill: dashed ? "#fff9ed" : color, stroke: color, "stroke-width": 1.7 }); dot.append(svgNode("title", {}, axes[i].name + "：" + displayNumber(value) + (dashed ? "（仅已确认问题，参考）" : "（规则与模型检出画像）"))); svg.append(dot); });
    }
    drawProfile(axes.map((a) => a.confirmedQualityScore), "#c1934d", true, "#c1934d0c");
    drawProfile(axes.map(axisDisplayScore), "#356cbe", false, "#356cbe12");
    $("quality-radar").replaceChildren(svg);
  }
  function renderSixDimensions(dimensions) {
    const names = { completeness: "完整性", annotation_quality: "标注质量", annotationQuality: "标注质量", sample_correspondence: "样本对应", sampleCorrespondence: "样本对应", coverage_distribution: "覆盖与分布", coverageDistribution: "覆盖与分布", documentation_provenance: "文档与来源追溯", traceability: "文档与来源追溯", usage_conditions: "使用条件", usageConditions: "使用条件" };
    $("six-dimension-grid").replaceChildren(); const rows = Array.isArray(dimensions) ? dimensions : Object.entries(dimensions).map(([id, row]) => ({ id, ...row }));
    for (const row of rows) {
      const card = element("div", "dimension-card"), score = finite(row.displayScore) ? row.displayScore : row.score;
      card.append(element("strong", "", names[row.id] || row.name || row.id), element("span", "", finite(score) ? displayNumber(score) + "分 · " + (row.scoreLabel || "已核验范围") : row.statusLabel || "需补充判定依据"));
      const coverage = finite(row.scopeCheckCoverage) ? row.scopeCheckCoverage : row.checkCoverage;
      if (finite(coverage)) card.append(element("small", "", (row.scoreKind ? "限定指标完成 " : "可核验检查覆盖 ") + displayNumber(coverage * 100, "%")));
      if (row.scopeCounts?.undetermined && Array.isArray(row.scoreBounds)) card.append(element('small','',`含未决证据的区间 ${displayNumber(row.scoreBounds[0])}–${displayNumber(row.scoreBounds[1])}分`));
      card.append(element("p", "", String(row.reason || row.meaning || row.state || "详见评价规范")));
      if (row.criteria?.length) {
        const detail=element('details','six-check-details');detail.append(element('summary','','查看计分依据与证据'));
        for(const check of row.criteria){const counts=check.counts||{},done=(counts.pass||0)+(counts.fail||0);detail.append(element('p','',`${check.checkId} ${check.name}：通过${counts.pass||0} / 已核验${done}`));}
        card.append(detail);
      }
      $("six-dimension-grid").append(card);
    }
    if (!rows.length) $("six-dimension-grid").append(element("p", "panel-empty", "当前报告未提供六维覆盖明细。"));
  }
  function showTier(tier = "clean") {
    workspace.tier = tier;
    workspace.offset = 0;
    $("case-defect-filter").value = tier === "clean" ? "" : tier;
    $("case-stage-filter").value = "";
    const entry = (workspace.insights?.tiers || []).find((item) => item.id === tier);
    $("cases-title").textContent = `${entry?.name || (tier === "clean" ? "干净样本" : defectName(tier))} · 样本列表`;
    setWorkspace("cases");
  }
  function showDefectCases(code = "") { showTier(code || "clean"); }
  async function loadCases() {
    const id = state.activeId; if (!id) return;
    workspace.casesController?.abort(); const controller = new AbortController(); workspace.casesController = controller; const token = ++workspace.casesToken;
    const query = new URLSearchParams({ offset: String(workspace.offset), limit: String(workspace.limit) });
    for (const [name, value] of [["defectType", $("case-defect-filter").value], ["stage", $("case-stage-filter").value], ["q", $("case-search").value.trim()]]) if (value) query.set(name, value);
    if (workspace.tier) query.set("tier", workspace.tier);
    $("case-list").replaceChildren(element("p", "panel-empty", "正在读取真实案例…")); $("case-results-caption").textContent = "正在检索案例记录";
    $("cases-prev").disabled = true; $("cases-next").disabled = true;
    try {
      const endpoint = workspace.tier ? "tiers" : "cases";
      const result = await request(`/api/chat/sessions/${encodeURIComponent(id)}/${endpoint}?${query}`, { signal: controller.signal, timeout: 45000 });
      if (controller.signal.aborted || token !== workspace.casesToken || state.activeId !== id) return;
      const items = Array.isArray(result.items) ? result.items : []; workspace.total = Number(result.total) || 0;
      $("case-results-caption").textContent = workspace.tier ? `共 ${workspace.total.toLocaleString("zh-CN")} 个${result.tierName || "档案"}样本；图像和 CSV 预览直接读取服务器原文件。` : `共 ${workspace.total.toLocaleString("zh-CN")} 条缺陷案例记录；同一样本可有多个缺陷，不等于问题样本总数。`;
      $("case-count-badge").textContent = workspace.total ? displayNumber(workspace.total) : ""; $("case-list").replaceChildren();
      for (const item of items) {
        const card = element("button", "case-card"); card.type = "button";
        const top = element("span", "case-card-top"); top.append(element("span", `case-stage ${item.stage === "confirmed" ? "confirmed" : item.stage === "clean" ? "clean" : "candidate"}`, stageName(item.stage)), element("span", "case-code", item.defectType === "CLEAN" ? "CLEAN" : item.defectType));
        const preview = localUrl(item.previewUrl);
        if (preview) { const image = document.createElement("img"); image.className = "case-card-preview"; image.loading = "lazy"; image.alt = item.dataName || item.sampleName || "样本预览"; image.src = preview; card.append(image); }
        else if (item.previewKind === "timeseries") card.append(element("div", "case-card-preview series-placeholder", "CSV 波形"));
        card.append(top, element("strong", "case-card-defect", item.defectName || defectName(item.defectType) || (item.defectType === "CLEAN" ? "干净样本" : item.defectType)), element("span", "case-card-name", item.dataName || item.labelName || item.sampleName || item.sampleId || item.caseId), element("span", "case-card-basis", item.basis || "打开样本查看原图、标注与分析依据"));
        const bottom = element("span", "case-card-bottom"), location = element("span", "", caseLocationLabel(item));
        if (location.textContent === "含时段定位") location.title = "时段来自报告中的原始记录；具体证据与确认状态见案例分析，不代表已知独立真实端点。";
        bottom.append(location, element("span", "", "查看分析 ↗")); card.append(bottom); card.addEventListener("click", () => openCase(item.caseId)); $("case-list").append(card);
      }
      if (!items.length) $("case-list").append(element("p", "panel-empty", "当前筛选没有案例。可以清空搜索或切换缺陷类型。"));
      $("cases-prev").disabled = workspace.offset <= 0; $("cases-next").disabled = workspace.offset + items.length >= workspace.total;
      $("case-page-label").textContent = workspace.total ? `${Math.floor(workspace.offset / workspace.limit) + 1} / ${Math.ceil(workspace.total / workspace.limit)} 页` : "0 条记录";
    } catch (error) {
      if (controller.signal.aborted || token !== workspace.casesToken || state.activeId !== id) return;
      const retry = element("button", "text-button", "重新读取案例 →"); retry.type = "button"; retry.addEventListener("click", loadCases); $("case-list").replaceChildren(element("p", "panel-empty", errorMessage(error)), retry); $("case-results-caption").textContent = "案例暂未载入";
    }
  }
  function closeCase() { workspace.detailToken++; workspace.detailController?.abort(); workspace.explainController?.abort(); if ($("case-dialog").open) $("case-dialog").close(); }
  async function openCase(caseId) {
    if (!caseId || !state.activeId) return;
    const id = state.activeId, token = ++workspace.detailToken; workspace.detailController?.abort(); workspace.explainController?.abort();
    const controller = new AbortController(); workspace.detailController = controller; workspace.detail = null;
    $("case-dialog-title").textContent = "正在读取案例…"; $("case-detail-loading").hidden = false; $("case-detail-loading").textContent = "正在读取原始样本、定位与逐案分析…"; $("case-detail-content").hidden = true;
    $("case-llm-answer").hidden = true; $("case-llm-answer").textContent = ""; $("case-llm-source").textContent = ""; $("case-explain").disabled = false;
    if (!$("case-dialog").open) $("case-dialog").showModal();
    try {
      const detail = await request(`/api/chat/sessions/${encodeURIComponent(id)}/cases/${encodeURIComponent(caseId)}`, { signal: controller.signal, timeout: 45000 });
      if (controller.signal.aborted || token !== workspace.detailToken || state.activeId !== id) return;
      workspace.detail = detail; renderCase(detail);
      const preview = localUrl(detail.sample?.imageUrl);
      if (preview) {
        const image = new Image(); image.src = preview; let timer;
        try { await Promise.race([image.decode(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error("原图加载超时，请重新读取案例。")), 20000); })]); }
        finally { clearTimeout(timer); }
      }
      if (controller.signal.aborted || token !== workspace.detailToken || state.activeId !== id) return;
      $("case-detail-loading").hidden = true; $("case-detail-content").hidden = false;
    } catch (error) {
      if (controller.signal.aborted || token !== workspace.detailToken || state.activeId !== id) return;
      const retry = element("button", "text-button", "重新读取 →"); retry.type = "button"; retry.addEventListener("click", () => openCase(caseId)); $("case-detail-loading").replaceChildren(element("p", "", errorMessage(error)), retry);
    }
  }
  function renderCase(detail) {
    const item = detail.case || {}, sample = detail.sample, analysis = detail.analysis || {};
    $("case-dialog-kicker").textContent = `${item.defectType || "缺陷"} · ${stageName(item.stage)}`; $("case-dialog-title").textContent = item.defectName || defectName(item.defectType);
    $("case-detail-badges").replaceChildren(element("span", `case-stage ${item.stage === "confirmed" ? "confirmed" : item.stage === "clean" ? "clean" : "candidate"}`, stageName(item.stage)), element("span", "case-sample-id", item.dataName || item.labelName || item.sampleName || item.sampleId || item.caseId));
    $("case-label-name").textContent = sample?.labelName || item.labelName || ""; $("case-label-json").textContent = sample?.label ? JSON.stringify(sample.label, null, 2).slice(0, 500000) : "当前未读到原始标注；仍保留报告中的样本定位与判断依据。";
    $("case-evidence-json").textContent = JSON.stringify({ caseId: item.caseId, sampleId: item.sampleId, defectType: item.defectType, checkId: item.checkId, ruleVersion: item.ruleVersion, stage: item.stage, scoreEligible: item.scoreEligible, sourceKind: item.sourceKind, locator: item.locator, confidence: item.confidence, severity: item.severity }, null, 2);
    $("case-analysis-sections").replaceChildren();
    for (const [title, value] of [["结论", analysis.summary || item.basis], ["判断依据", analysis.reasoning || item.basis], ["问题位置", analysis.location || item.locator], ["偏差与程度", analysis.severity || item.severity], ["影响", analysis.impact], ["复核建议", analysis.recommendation], ["结论状态", analysis.statusMeaning]]) {
      if (value === undefined || value === null || value === "") continue; const section = element("section", "case-analysis-section"); section.append(element("h4", "", title)); if (Array.isArray(value)) { const list = element("ul"); for (const line of value) list.append(element("li", "", textValue(line))); section.append(list); } else section.append(element("p", "", textValue(value))); $("case-analysis-sections").append(section);
    }
    if (analysis.expertContributions?.length) { const section = element("section", "case-analysis-section expert-contributions"); section.append(element("h4", "", "实际工具与专家依据")); for (const expert of analysis.expertContributions) { const line = element("p", ""); line.append(element("strong", "", expert.name || "检查工具"), document.createTextNode(" · " + (expert.role || "") + (finite(expert.probability) ? " · 概率 " + displayNumber(expert.probability * 100, "%") : "")), element("small", "", textValue(expert.evidence || ""))); section.append(line); } $("case-analysis-sections").append(section); }
    $("case-measurements").replaceChildren();
    if (item.measurements && Object.keys(item.measurements).length) { const section = element("details", "measurement-details"); section.open = true; section.append(element("summary", "", "实测值与声明值"), element("pre", "", JSON.stringify(item.measurements, null, 2))); $("case-measurements").append(section); }
    $("toggle-input-annotations").checked = true; $("toggle-defect-location").checked = true;
    drawCaseVisual(detail);
  }
  function drawCaseVisual(detail) {
    const item = detail.case || {}, sample = detail.sample, visual = $("case-visual"); visual.replaceChildren();
    const image = sample && localUrl(sample.imageUrl), width = sample?.width, height = sample?.height;
    $("toggle-input-annotations").disabled = !image || !sample?.annotations?.length; $("toggle-defect-location").disabled = !image || !Array.isArray(item.locator?.bbox);
    if (image && finite(width) && finite(height) && width > 0 && height > 0) {
      $("case-preview-title").textContent = `原始图像 · ${width} × ${height}`;
      const svg = svgNode("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "原图与可分别开关的输入标注、问题定位" }); svg.append(svgNode("image", { href: image, width, height }));
      const inputGroup = svgNode("g", { class: "input-annotation-layer" });
      for (const shape of sample.annotations || []) { const points = (shape.points || []).filter((p) => Array.isArray(p) && p.length >= 2 && finite(p[0]) && finite(p[1])); if (points.length < 3) continue; const polygon = svgNode("polygon", { points: points.map((p) => p.slice(0, 2).join(",")).join(" "), fill: "#33c5ad15", stroke: "#33c5ad", "stroke-width": Math.max(width, height) / 350 }); polygon.append(svgNode("title", {}, "输入标注：" + (shape.label || "目标") + " " + (shape.pointer || ""))); inputGroup.append(polygon); } svg.append(inputGroup);
      const defectGroup = svgNode("g", { class: "defect-location-layer" }), bbox = item.locator?.bbox;
      if (Array.isArray(bbox) && bbox.length === 4 && bbox.every(finite) && bbox[2] > bbox[0] && bbox[3] > bbox[1] && (!item.locator.coordinate_system || item.locator.coordinate_system === "image_pixels")) { defectGroup.append(svgNode("rect", { x: bbox[0], y: bbox[1], width: bbox[2] - bbox[0], height: bbox[3] - bbox[1], fill: "#ed7f6318", stroke: "#ff7d63", "stroke-width": Math.max(width, height) / 230, "stroke-dasharray": `${Math.max(width, height) / 60} ${Math.max(width, height) / 120}` })); } svg.append(defectGroup); visual.append(svg);
      $("case-visual-caption").textContent = state.current?.dataset?.family === 'EPB_seg_sample' && Array.isArray(bbox) ? '青绿色为输入分割多边形，橙红矩形仅表示定位候选范围，尚无独立预测的分割轮廓。两层可分别开关。' : item.locator?.region_source === "questioned_input_annotation_not_independent_predicted_defect_region" ? "青绿色为输入标注，橙红虚线指出被质疑的输入标注对象；这是待复核对象的位置，尚不能确认虚标。两层可分别开关。" : "青绿色为输入标注，橙红虚线为报告中的问题定位；两层可分别开关。没有区域定位的字段问题不绘制猜测框。";
    } else if (sample?.series?.length) {
      $("case-preview-title").textContent = "原始 CSV 波形 · 行号坐标"; drawSeries(sample.series, visual);
      $("case-visual-caption").textContent = `预览 ${sample.series.length} 个数值通道，共 ${displayNumber(sample.totalRows)} 行。各通道独立纵轴，横轴均为源行号；通道单位可能不同，不直接比较幅值。事件起止时刻见右侧定位文本，不叠到行号轴。`;
    } else {
      $("case-preview-title").textContent = sample?.missingData ? "对应数据文件缺失" : "样本预览";
      const empty = element("div", "missing-preview"); empty.append(element("span", "", sample?.missingData || item.defectType === "L0-01" ? "▱" : "◇"), element("strong", "", sample?.missingData || item.defectType === "L0-01" ? "标注保留，原始数据文件缺失" : "当前没有可显示的图像或曲线"), element("p", "", detail.sampleError || (sample?.warnings || []).join("；") || "可在下方查看原始标注，并核对报告中的预期文件与样本路径。")); visual.append(empty);
      $("case-visual-caption").textContent = "空白预览保留为缺失证据，不以示例图替代原数据。";
    }
    updateVisualLayers();
  }
  function updateVisualLayers() { document.querySelectorAll(".input-annotation-layer").forEach((g) => g.style.display = $("toggle-input-annotations").checked ? "" : "none"); document.querySelectorAll(".defect-location-layer").forEach((g) => g.style.display = $("toggle-defect-location").checked ? "" : "none"); }
  function drawSeries(series, target) {
    const colors = ["#39786d", "#c1934d", "#8584b5", "#6491ad", "#c47a6e", "#9ba85e"], rows = series.slice(0, 6).map((row) => ({ ...row, points: (row.points || []).filter((p) => Array.isArray(p) && p.length >= 2 && finite(p[0]) && finite(p[1])).slice(0, 2400) })).filter((row) => row.points.length);
    if (!rows.length) { target.append(element("p", "panel-empty", "CSV未提供可绘制的有限数值。")); return; }
    const all = rows.flatMap((row) => row.points); let minX = Math.min(...all.map((p) => p[0])), maxX = Math.max(...all.map((p) => p[0])); if (maxX === minX) maxX++;
    const svg = svgNode("svg", { viewBox: `0 0 700 ${rows.length * 170 + 25}`, role: "img", class: "series-chart", "aria-label": "CSV各通道独立纵轴曲线，统一横坐标为原始行号" }), x = (value) => 70 + (value - minX) / (maxX - minX) * 605;
    rows.forEach((row, index) => {
      let minY = Math.min(...row.points.map((p) => p[1])), maxY = Math.max(...row.points.map((p) => p[1])); if (maxY === minY) { minY--; maxY++; }
      const top = index * 170 + 30, bottom = top + 110, y = (value) => bottom - (value - minY) / (maxY - minY) * 110;
      svg.append(svgNode("text", { x: 70, y: top - 11, fill: colors[index], "font-size": 12 }, row.name || `通道 ${index + 1}`));
      for (let i = 0; i < 3; i++) { const val = minY + (maxY - minY) * i / 2, yy = y(val); svg.append(svgNode("line", { x1: 70, y1: yy, x2: 675, y2: yy, stroke: "#dfe5d8", "stroke-dasharray": "4 4" }), svgNode("text", { x: 62, y: yy + 3, "text-anchor": "end", fill: "#929e87", "font-size": 9 }, Number(val.toPrecision(4)).toLocaleString())); }
      svg.append(svgNode("polyline", { points: row.points.map((p) => `${x(p[0])},${y(p[1])}`).join(" "), fill: "none", stroke: colors[index], "stroke-width": 1.4 }), svgNode("text", { x: 70, y: bottom + 15, fill: "#929e87", "font-size": 9 }, displayNumber(minX)), svgNode("text", { x: 675, y: bottom + 15, "text-anchor": "end", fill: "#929e87", "font-size": 9 }, displayNumber(maxX)));
    });
    svg.append(svgNode("text", { x: 368, y: rows.length * 170 + 10, "text-anchor": "middle", fill: "#7d8c70", "font-size": 11 }, "源文件行号（各通道独立纵轴）")); target.append(svg);
    const legend = element("div", "series-legend"); rows.forEach((row, i) => { const item = element("span", "", row.name || `通道 ${i + 1}`); item.style.color = colors[i]; legend.append(item); }); target.append(legend);
  }
  async function explainCase() {
    const item = workspace.detail?.case; if (!item?.caseId || !state.activeId) return;
    const id = state.activeId, token = workspace.detailToken, controller = new AbortController(); workspace.explainController?.abort(); workspace.explainController = controller;
    $("case-explain").disabled = true; $("case-explain").textContent = "正在依据本案例组织解释…"; $("case-llm-answer").hidden = false; $("case-llm-answer").textContent = "正在读取这个案例的证据并组织说明，请稍候。"; $("case-llm-source").textContent = "";
    try {
      const result = await request(`/api/chat/sessions/${encodeURIComponent(id)}/cases/${encodeURIComponent(item.caseId)}/explain`, { method: "POST", body: JSON.stringify({ message: "请用200字以内解释这个案例是什么问题、证据在哪里、偏差有多大、可能影响和复核建议，保持确认或待复核状态。" }), signal: controller.signal, timeout: 60000 });
      if (controller.signal.aborted || token !== workspace.detailToken || id !== state.activeId) return;
      $("case-llm-answer").textContent = String(result.content || "当前没有可显示的解释，请查看上方逐案分析。").replaceAll("待评估", "需补充依据");
      $("case-llm-source").textContent = `${result.replySource === "grounded_llm" ? result.model || "语言模型解释" : "依据实际记录整理"}${result.cached ? " · 复用已保存的真实解释" : ""} · 保留原结论状态`;
    } catch (error) { if (!controller.signal.aborted && token === workspace.detailToken) $("case-llm-answer").textContent = errorMessage(error) + " 上方基于证据的逐案分析仍可使用，可再次点击重试。"; }
    finally { if (token === workspace.detailToken) { $("case-explain").disabled = false; $("case-explain").textContent = "让大模型解释此案例 ✳"; } }
  }
  function downloadRadar() {
    const source = $("quality-radar").querySelector("svg"); if (!source) return;
    const svg = source.cloneNode(true); svg.setAttribute("xmlns", "http://www.w3.org/2000/svg"); svg.setAttribute("viewBox", "0 -45 580 565"); svg.setAttribute("width", "1160"); svg.setAttribute("height", "1130");
    svg.prepend(svgNode("rect", { x: 0, y: -45, width: 580, height: 565, fill: "#fcfcf7" }));
    svg.append(svgNode("text", { x: 290, y: -19, "text-anchor": "middle", "font-size": 16, fill: "#354c3c" }, "SpaceQA · 八类检出画像分"));
    svg.append(svgNode("text", { x: 290, y: 456, "text-anchor": "middle", "font-size": 10, fill: "#708166" }, datasetName(state.current)));
    svg.append(svgNode("text", { x: 290, y: 478, "text-anchor": "middle", "font-size": 9, fill: "#39786d" }, "青绿：规则与模型检出画像　橙色：仅确认参考　灰色：未覆盖或不适用"));
    svg.append(svgNode("text", { x: 290, y: 497, "text-anchor": "middle", "font-size": 9, fill: "#8a927f" }, "0–100分，越高表示当前检出问题越少；八类画像不等同官方六维评分。"));
    const blob = new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml;charset=utf-8" }); const url = URL.createObjectURL(blob), link = document.createElement("a"); link.href = url; link.download = "SpaceQA_" + (state.current?.dataset?.name || "dataset").replace(/[^\w\u4e00-\u9fff.-]/g, "_") + "_eight_axes.svg"; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  for (const [code, name] of DEFECTS) $("case-defect-filter").append(new Option(name + " · " + code, code));
  for (const view of ["overview", "cases", "conversation"]) $("view-" + view).addEventListener("click", () => setWorkspace(view));
  $("insights-refresh").addEventListener("click", async () => { await fetchActive(); await loadInsights(); });
  $("live-run-button").addEventListener("click", startLiveRun);
  $("download-radar").addEventListener("click", downloadRadar);
  $("show-all-cases").addEventListener("click", () => { workspace.tier = ""; workspace.offset = 0; $("case-defect-filter").value = ""; $("case-stage-filter").value = ""; $("cases-title").textContent = "问题案例库"; setWorkspace("cases"); });
  $("cases-refresh").addEventListener("click", loadCases);
  for (const id of ["case-defect-filter", "case-stage-filter"]) $(id).addEventListener("change", () => { workspace.tier = ""; $("cases-title").textContent = "问题案例库"; workspace.offset = 0; loadCases(); });
  $("case-search").addEventListener("input", () => { clearTimeout(workspace.searchTimer); workspace.searchTimer = setTimeout(() => { workspace.offset = 0; loadCases(); }, 300); });
  $("cases-prev").addEventListener("click", () => { workspace.offset = Math.max(0, workspace.offset - workspace.limit); loadCases(); });
  $("cases-next").addEventListener("click", () => { workspace.offset += workspace.limit; loadCases(); });
  $("tier-sample-close").addEventListener("click", () => { $("tier-sample-panel").hidden = true; });
  $("case-dialog-close").addEventListener("click", closeCase);
  $("case-dialog").addEventListener("cancel", () => { workspace.detailToken++; workspace.detailController?.abort(); workspace.explainController?.abort(); });
  $("toggle-input-annotations").addEventListener("change", updateVisualLayers); $("toggle-defect-location").addEventListener("change", updateVisualLayers);
  $("case-explain").addEventListener("click", explainCase);
  $("case-ask-chat").addEventListener("click", () => { const item = workspace.detail?.case; closeCase(); setWorkspace("conversation"); $("message-input").value = item ? `请说明${item.defectName || defectName(item.defectType)}这类问题的判断依据与复核方法。` : "请说明这个包的问题依据和复核方法。"; resizeInput(); $("message-input").focus(); });

  $("new-session").addEventListener("click", () => openDatasetDialog());
  $("home-link").addEventListener("click", (event) => { if (event.button || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return; event.preventDefault(); showWelcome(); });
  $("welcome-select").addEventListener("click", () => openDatasetDialog());
  $("welcome-upload").addEventListener("click", () => openDatasetDialog("upload"));
  $("composer-attach").addEventListener("click", () => openDatasetDialog());
  $("refresh-sessions").addEventListener("click", () => loadSessions(false));
  $("notice-action").addEventListener("click", () => state.noticeAction?.());
  $("notice-close").addEventListener("click", hideNotice);
  $("message-form").addEventListener("submit", sendMessage);
  $("message-input").addEventListener("input", resizeInput);
  $("message-input").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendMessage(); } });
  $("retry-analysis").addEventListener("click", retryAnalysis);
  $("mobile-menu").addEventListener("click", () => $("sidebar").classList.contains("open") ? closeSidebar() : openSidebar());
  $("sidebar-scrim").addEventListener("click", closeSidebar);
  document.addEventListener("keydown", (event) => {
    if (!$("sidebar").classList.contains("open") || $("dataset-dialog").open) return;
    if (event.key === "Escape") { event.preventDefault(); closeSidebar(); return; }
    if (event.key !== "Tab") return;
    const focusable = [...$("sidebar").querySelectorAll('a[href],button:not([disabled]),[tabindex="0"]')].filter((node) => node.getClientRects().length);
    const first = focusable[0], last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  });
  matchMedia("(max-width: 760px)").addEventListener("change", (event) => { if (!event.matches) closeSidebar(); });
  $("dialog-close").addEventListener("click", closeDatasetDialog);
  $("dataset-dialog").addEventListener("cancel", (event) => { if (state.upload) event.preventDefault(); });
  $("dataset-dialog").addEventListener("click", (event) => { if (event.target !== $("dataset-dialog")) return; const rect = $("dataset-dialog").getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) closeDatasetDialog(); });
  $("tab-existing").addEventListener("click", () => setTab("existing"));
  $("tab-upload").addEventListener("click", () => setTab("upload"));
  $("dataset-search").addEventListener("input", renderFolders);
  $("family-filter").addEventListener("change", renderFolders);
  $("refresh-folders").addEventListener("click", () => browseFolder(state.folderPath));
  $("add-selected-dataset").addEventListener("click", () => createSession(state.selectedDataset));
  $("demo-dataset").addEventListener("click", () => { if (state.demo) createSession(state.demo.dataset); });
  $("demo-dataset").addEventListener("dragstart", (event) => { if (!state.demo || state.creating) { event.preventDefault(); return; } event.dataTransfer.setData("application/x-spaceqa-dataset", JSON.stringify(state.demo.dataset)); event.dataTransfer.effectAllowed = "copy"; $("dataset-drop-hint").hidden = false; });
  $("demo-dataset").addEventListener("dragend", () => { $("dataset-drop-hint").hidden = true; $("conversation-scroll").classList.remove("drop-ready"); });
  $("conversation-scroll").addEventListener("dragover", (event) => { if (!event.dataTransfer.types.includes("application/x-spaceqa-dataset")) return; event.preventDefault(); event.dataTransfer.dropEffect = "copy"; $("dataset-drop-hint").hidden = false; $("conversation-scroll").classList.add("drop-ready"); });
  $("conversation-scroll").addEventListener("drop", (event) => { const raw = event.dataTransfer.getData("application/x-spaceqa-dataset"); if (!raw) return; event.preventDefault(); $("dataset-drop-hint").hidden = true; $("conversation-scroll").classList.remove("drop-ready"); try { const dataset = JSON.parse(raw); if (typeof dataset.id === "string" && state.demo?.dataset.id === dataset.id) createSession(state.demo.dataset); } catch (_) { showNotice("无法读取拖入的数据集，请点击示例卡重试。"); } });
  $("drop-zone").addEventListener("click", () => $("zip-input").click());
  $("zip-input").addEventListener("change", () => uploadZip($("zip-input").files[0]));
  $("cancel-upload").addEventListener("click", () => state.upload?.abort());
  for (const name of ["dragenter", "dragover"]) $("drop-zone").addEventListener(name, (event) => { event.preventDefault(); $("drop-zone").classList.add("drag-over"); });
  for (const name of ["dragleave", "drop"]) $("drop-zone").addEventListener(name, (event) => { event.preventDefault(); $("drop-zone").classList.remove("drag-over"); });
  $("drop-zone").addEventListener("drop", (event) => { if (event.dataTransfer.files.length > 1) { dialogError("一次上传一个 ZIP 数据包，请分别添加。"); return; } uploadZip(event.dataTransfer.files[0]); });
  document.addEventListener("dragover", (event) => { if (event.dataTransfer?.types.includes("Files")) event.preventDefault(); });
  document.addEventListener("drop", (event) => { if (event.dataTransfer?.types.includes("Files")) event.preventDefault(); });
  document.querySelectorAll("[data-suggestion]").forEach((button) => button.addEventListener("click", () => { $("message-input").value = button.dataset.suggestion; resizeInput(); if (!state.activeId) openDatasetDialog(); else $("message-input").focus(); }));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stopPoll(); else if (state.activeId) fetchActive(); });
  window.addEventListener("online", () => { if (state.activeId) fetchActive(); else loadSessions(true); });
  window.addEventListener("offline", () => { connection(false); stopPoll(); showNotice("网络已断开，暂时无法获取最新状态。", () => state.activeId ? fetchActive() : loadSessions(true), "重新连接"); });
  window.addEventListener("beforeunload", () => { stopPoll(); state.readController?.abort(); });
  loadSessions(true);
  loadDemo();
})();
