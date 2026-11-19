/* Shared, dependency-free UI locale layer.
 *
 * The evaluator must keep raw dataset names, JSON, model answers and evidence
 * untouched. Only system UI text is translated. Each page declares its source
 * language with <body data-locale-source="en|zh">. The observer also covers
 * dynamically rendered tables, dialogs and job cards.
 */
(function () {
  "use strict";

  const pairs = [
    ["WORKSPACE", "工作空间"], ["Overview", "总览"], ["Dataset explorer", "数据集浏览"],
    ["Evaluation results", "评价结果"], ["Run an evaluation", "启动评价"], ["Evidence first", "以证据为基础"],
    ["Traceable checks. Reproducible scores. Original data stays unchanged.", "检查可追溯，分数可复算，保留原始数据。"],
    ["Project quality scores are not official competition F1 or scientific ground truth.", "项目质量分不代表官方 F1 或科学真值。"],
    ["Connecting…", "连接中…"], ["Connecting to the server…", "正在连接服务器…"], ["Connection unavailable", "连接不可用"],
    ["Refresh current view", "刷新当前页面"], ["Close detail", "关闭详情"], ["Your data, with evidence.", "让数据结论都有证据。"],
    ["Browse server datasets, evaluate quality and inspect every conclusion.", "浏览服务器数据集、评价质量并检查每条结论。"],
    ["＋ New evaluation", "＋ 新建评价"], ["Available datasets", "可用数据集"], ["Server-side packages", "服务器数据包"],
    ["Recorded evaluations", "已有评价"], ["Existing auditable outputs", "已有可审计结果"], ["Active jobs", "运行中任务"],
    ["No background evaluations", "没有后台评价任务"], ["Queued or currently running", "排队或正在运行"],
    ["Evaluation modes", "评价模式"], ["Available on this server", "本服务器可用"], ["Recent evaluations", "最近评价"],
    ["Open a run to inspect scores, findings and source evidence.", "打开一次运行，查看评分、问题和来源证据。"],
    ["View all →", "查看全部 →"], ["Dataset families", "数据集类型"], ["SERVER INVENTORY", "服务器清单"],
    ["No datasets configured", "尚未配置数据集"], ["What this console can verify", "本工作台可以核验"],
    ["Original data and annotation previews", "原始数据与标注预览"],
    ["Inspect provided polygons, image metadata and sampled time series.", "查看提供的多边形、图像元数据和抽样时间序列。"],
    ["Deterministic evaluation and output audit", "确定性评价与输出审计"],
    ["Run supported checks, follow job logs and recalculate project scores.", "执行支持的检查、跟踪任务日志并复算项目分数。"],
    ["Separate confirmed findings from pending candidates", "区分已确认问题与待复核候选"],
    ["Review provenance, locator type and versions before drawing conclusions.", "在下结论前核对来源、定位类型和版本。"],
    ["Execution boundaries", "执行边界"], ["Live expert inference", "实时专家推理"], ["Available", "可用"], ["Not enabled", "未启用"],
    ["External model API", "外部模型 API"], ["Not called by this console", "本工作台未调用"],
    ["Official roadshow F1", "路演官方 F1"], ["Not available without ground truth", "缺少真值，无法计算"],
    ["Source data handling", "源数据处理"], ["Read only", "只读"], ["Host CPU threads / load (1m)", "主机 CPU 线程 / 负载（1 分钟）"],
    ["Host memory available", "可用主机内存"], ["Result storage free", "结果存储剩余空间"],
    ["Host snapshot, not per-job resource peaks.", "这是主机快照，不是单个任务的资源峰值。"],
    ["Read results in context.", "请结合上下文阅读结果。"],
    ["Pending model candidates are separate from confirmed findings.", "模型候选与已确认问题分开统计。"],
    ["Cached integrations do not establish fresh GPU inference speed or official scientific accuracy.", "缓存的专家结果不能证明新的 GPU 推理速度或官方科学准确率。"],
    ["Explore the source data.", "浏览源数据。"], ["Select a package and inspect its original data alongside provided annotations.", "选择数据包，查看原始数据与提供的标注。"],
    ["DATA PACKAGES", "数据包"], ["Filter packages…", "筛选数据包…"], ["Filter dataset packages", "筛选数据包"],
    ["Loading samples…", "正在读取样本…"], ["No dataset packages found", "没有找到数据包"],
    ["The service needs a configured server data root.", "服务需要配置服务器数据根目录。"], ["Search sample filenames…", "搜索样本文件名…"],
    ["Search sample filenames", "搜索样本文件名"], ["Search", "搜索"], ["Original annotations · no automatic corrections", "原始标注 · 不自动修正"],
    ["Loading sample list…", "正在读取样本列表…"], ["No matching samples", "没有匹配的样本"],
    ["Try another search or select a different package.", "请尝试其他搜索词或选择其他数据包。"], ["Missing data", "数据文件缺失"],
    ["label available", "有标注"], ["no label", "无标注"], ["Evaluation results", "评价结果"],
    ["From scores to source evidence.", "从分数回到来源证据。"],
    ["Review saved outputs, pending candidates and the observations behind each finding.", "查看已保存结果、待复核候选及每条问题背后的观测。"],
    ["SAVED EVALUATIONS", "已保存评价"], ["No runs recorded", "没有已记录的运行"], ["No evaluation output yet", "还没有评价输出"],
    ["Use “Run an evaluation” to scan a package.", "使用“启动评价”扫描一个数据包。"], ["SAVED EVALUATION", "已保存评价"],
    ["Audit passed", "审计通过"], ["Audit needs review", "审计需要复核"], ["Audit not recorded", "未记录审计"],
    ["Packages", "数据包数"], ["In this evaluation", "本次评价"], ["Confirmed issues", "已确认问题"],
    ["Technical findings, not official TP", "技术发现，不是官方 TP"], ["Pending review", "待复核"], ["Excluded from confirmed penalties", "不计入已确认扣分"],
    ["Pass records", "通过记录"], ["Check-level observations", "检查级观测"], ["Scorecards", "评分卡"], ["Ranking", "排序"],
    ["Findings & evidence", "问题与证据"], ["Audit & logs", "审计与日志"], ["Project specification scores.", "项目规范分。"],
    ["No scorecards found", "没有找到评分卡"], ["No ranking table available", "没有可用排序表"], ["No matching findings", "没有匹配的问题"],
    ["Clear the filters or inspect another evaluation.", "清除筛选条件或查看其他评价。"], ["Run manifest", "运行清单"],
    ["No saved audit record", "没有保存的审计记录"], ["Recorded run log", "已记录运行日志"], ["SERVER EXCERPT", "服务器日志摘录"],
    ["No run log available.", "没有可用运行日志。"], ["Launch a reproducible evaluation.", "启动可复现的评价。"],
    ["Select an existing server package. Source files remain unchanged; results go to a new run directory.", "选择服务器上的数据包。源文件保持不变，结果写入新的运行目录。"],
    ["New evaluation", "新建评价"], ["SERVER-SIDE JOB", "服务器任务"], ["Dataset package", "数据包"],
    ["Select a server dataset…", "选择服务器数据集…"], ["Only configured server datasets can be submitted.", "只能提交已配置的服务器数据集。"],
    ["Evaluation mode", "评价模式"], ["Random seed", "随机种子"], ["Scan → output → audit.", "扫描 → 输出 → 审计。"],
    ["▷ Start evaluation", "▷ 启动评价"], ["How evaluation works", "评价如何工作"], ["Choose a package and a fixed seed.", "选择数据包和固定随机种子。"],
    ["The server selects applicable checks for its supported data structure.", "服务器根据数据结构选择适用的检查。"], ["Follow the real job status.", "跟踪真实任务状态。"],
    ["Queued and running stages, errors and cancellation remain visible.", "排队、运行、错误和取消状态都会保留。"], ["Open the completed evaluation.", "打开已完成的评价。"],
    ["Review scorecards, pending findings and source evidence.", "查看评分卡、待复核问题和来源证据。"], ["No jobs submitted", "尚未提交任务"],
    ["Start your first evaluation using the package selector above.", "使用上方数据包选择器启动第一次评价。"], ["Cancel", "取消"], ["Open results →", "打开结果 →"],
    ["Previous", "上一页"], ["Next →", "下一页 →"], ["of", "共"], ["records", "条记录"],
    ["Could not load this view.", "无法加载当前页面。"], ["Server returned", "服务器返回"], ["Unknown", "未知"], ["Not recorded", "未记录"], ["Not evaluated", "未评价"],
    ["Data Quality Console", "数据质量工作台"], ["Close detail", "关闭详情"], ["Detail", "详情"], ["SOURCE DATA", "源数据"],
    ["Sample preview", "样本预览"], ["FINDING & EVIDENCE", "问题与证据"], ["Finding / check", "问题 / 检查"], ["Class", "类别"],
    ["Status", "状态"], ["Observation", "观测"], ["All statuses", "全部状态"], ["All packages", "全部数据包"],
    ["All defect classes", "全部缺陷类别"], ["Apply", "应用"], ["Search finding, sample or description…", "搜索问题、样本或描述…"],
    ["Finding status", "问题状态"], ["Finding dataset", "问题数据集"], ["Defect class", "缺陷类别"], ["Recorded locator", "记录的定位"],
    ["Tool observation", "工具观测"], ["Full finding record", "完整问题记录"], ["Full evidence record", "完整证据记录"], ["Open referenced sample →", "打开引用样本 →"],
    ["Original label JSON", "原始标注 JSON"], ["Original data + provided annotation", "原始数据 + 提供的标注"],
    ["Show provided annotations", "显示提供的标注"], ["These overlays come from the input label. They are not repaired annotations or newly predicted defect boundaries.", "这些叠加层来自输入标注，不是修复后的标注，也不是新预测的问题边界。"],
    ["CSV time-series preview", "CSV 时间序列预览"], ["Sampled points · hover to inspect", "抽样点 · 悬停查看"], ["No displayable image or series", "没有可显示的图像或序列"],
    ["The label and metadata remain available below.", "下方仍保留标注和元数据。"], ["No finite numeric points", "没有有限数值点"],
    ["Locator not recorded", "未记录定位"], ["Check ID", "检查编号"], ["Rule version", "规则版本"], ["Dataset", "数据集"],
    ["Evidence ID", "证据编号"], ["Score eligible", "是否计分"], ["Evidence integrity", "证据完整性"], ["Verified", "已核验"],
    ["Not verified / not recorded", "未核验 / 未记录"], ["What this console can verify", "本工作台可以核验"],

    // Chat workspace (Chinese source strings).
    ["空间科学数据质量助手", "Space Science Data Quality Assistant"], ["返回SpaceQA欢迎首页，保留历史会话", "Return to SpaceQA home and keep previous sessions"],
    ["新建数据集对话", "New dataset chat"], ["最近的数据集", "Recent datasets"], ["刷新会话列表", "Refresh session list"],
    ["刷新列表", "Refresh list"], ["正在连接你的工作空间…", "Connecting to your workspace…"], ["连接中", "Connecting"],
    ["打开原分析工作台", "Open analysis console"], ["分析工作台 ↗", "Analysis console ↗"], ["打开会话列表", "Open session list"],
    ["关闭侧边栏", "Close sidebar"], ["模型连接状态", "Model connection status"], ["分析工作空间", "Analysis workspace"],
    ["让每一个质量结论都有据可查", "Every quality conclusion is traceable"], ["放下数据集，立即查看已有分析", "Drop a dataset to view its existing analysis"],
    ["保留报告来源和原始评价时间", "Keep the report source and original evaluation time"], ["从一个数据集，", "Start with one dataset,"],
    ["开始一次有依据的检查。", "begin an evidence-based check."], ["选择服务器上的数据包，或上传 ZIP。助手会自动分析数据与标注，你可以继续追问问题、位置和判断依据。", "Choose a server package or upload a ZIP. The assistant analyzes data and labels, then answers follow-up questions about issues, locations and evidence."],
    ["浏览服务器文件夹", "Browse server folders"], ["上传 ZIP 文件", "Upload ZIP file"], ["示例数据集", "Demo dataset"], ["拖入对话区", "Drag into chat"], ["或点击添加 ↗", "or click to add ↗"],
    ["你可以这样提问", "Try asking"], ["整体质量怎么样？", "How is the overall quality?"], ["查看检出比例与主要问题", "See detection rates and major issues"],
    ["问题具体在哪里？", "Where are the issues?"], ["回到样本、标注和图像区域", "Trace back to samples, labels and image regions"], ["这些结论可靠吗？", "How reliable are these conclusions?"],
    ["了解检查范围与判断依据", "Understand the scope and evidence"], ["原始数据与输入标注保留原貌。模型候选与已确认缺陷分别展示。", "Original data and input labels remain unchanged. Model candidates and confirmed defects are shown separately."],
    ["数据集分析视图", "Dataset analysis views"], ["质量总览", "Quality overview"], ["案例库", "Case library"], ["对话追问", "Ask follow-up"], ["路演12包排序 ↗", "Roadshow ranking ↗"],
    ["现场整包重评", "Live package evaluation"], ["LIVE EVALUATION", "现场评价"], ["真实执行，逐步可见", "Live execution, step by step"],
    ["规则核验 · 八类融合 · 独立图像定位 · 六维报告", "Rule checks · eight-defect fusion · independent image localization · six-dimension report"],
    ["现场重评整包 ▶", "Re-evaluate package ▶"], ["刷新结果 ↻", "Refresh results ↻"], ["这个数据集，质量如何？", "How is this dataset?"],
    ["从总体到证据", "From overview to evidence"], ["八类缺陷质量画像", "Eight-defect quality profile"], ["八类检出画像分", "Eight-defect detection profile"],
    ["Eight-class pollution index", "八类污染指数"], ["Accumulated above probability thresholds; lower is better. A sample may contribute to multiple classes.", "按概率阈值累计，指数越低越好；同一样本可在多个类别中重复计入。"],
    ["Not generated", "未生成"], ["Overall ", "综合 "], ["No reproducible probabilities", "当前无可复算概率"], ["Denominator", "统计分母"], [" registered samples", "个登记样本"], ["Ranking direction", "排序方向"], ["Pollution index ascending (lower is better)", "污染指数升序（越低越好）"], ["Threshold rule", "阈值规则"], ["Only probabilities at or above the class threshold count", "仅概率 ≥ 类别阈值计入"],
    ["Missing probability ", "缺概率 "], [" records", " 条"], ["Class probabilities or deterministic evidence available", "已有类别概率或确定性证据"], ["The pollution index is a rule/model observation; without independent truth it is not an official rate or recall.", "污染指数来自当前规则/模型概率观测，没有独立真值时不能解释为官方污染率或召回率。"],
    ["Localization tolerance and expert-fusion audit", "定位容差与专家融合审计"], ["Hits require an independent reference", "只在有独立参考时计命中"], ["Localization prerequisite", "定位前提"], ["Independent reference required; otherwise keep as candidate", "必须有独立参考；没有时只保留候选"], ["Box IoU", "框 IoU"], ["Center distance", "中心距离"], ["Polygon IoU", "多边形 IoU"], ["Time endpoint (seconds)", "时间端点（秒）"], ["Relative dimension error", "尺寸相对误差"], ["This report has no raw expert probabilities, so no outlier removals are fabricated. The audit lists excluded experts when raw outputs are available.", "本次报告没有保存专家原始概率，未虚构异常值剔除记录。启用融合审计后会在这里列出被剔除专家及原因。"], ["MAD audit: ", "MAD 审计："], [" outputs flagged as outliers", " 个输出被标记为异常"], ["Kept ", "保留 "], [" experts", " 个专家"], ["No expert exceeded the MAD threshold", "没有超过MAD阈值的专家"], ["Excluded: ", "剔除："],
    ["Six-dimension formal score: conditions met", "六维正式分：条件已满足"], ["Six-dimension formal score: cannot claim all criteria yet", "六维正式分：暂不能宣称全部达标"], ["All applicable checks, specification materials and implementation consistency are recorded.", "六维适用检查、规范材料和实现一致性均已记录。"],
    ["0–100分，越高越好", "0–100, higher is better"], ["下载八维图 SVG ↓", "Download eight-axis SVG ↓"], ["检出画像（规则＋模型）", "Detection profile (rules + models)"],
    ["仅已确认问题（参考）", "Confirmed issues only (reference)"], ["不适用或未覆盖：不补分", "Not applicable or uncovered: no imputation"],
    ["逐类统计与证据", "Per-defect statistics and evidence"], ["点击任一类，进入对应案例。不同缺陷可出现在同一样本。", "Select a defect to open its cases. One sample may contain multiple defects."],
    ["浏览全部案例 →", "Browse all cases →"], ["六维评价规范与检查覆盖", "Six-dimension specification and check coverage"], ["与八类缺陷分别展示", "Shown separately from the eight defects"],
    ["为什么有些检查无法判定？", "Why are some checks unresolved?"], ["查看原因与下一步", "See reasons and next steps"], ["录屏演示指引", "Recording guide"],
    ["四步展示完整链路", "Show the complete workflow in four steps"], ["问题案例库", "Issue case library"], ["刷新 ↻", "Refresh ↻"],
    ["全部缺陷类型", "All defect types"], ["全部状态", "All statuses"], ["已确认问题", "Confirmed issue"], ["待复核候选", "Review candidate"],
    ["搜索样本或判断依据…", "Search samples or evidence…"], ["搜索案例", "Search cases"], ["正在读取案例…", "Loading cases…"], ["上一页", "Previous"],
    ["下一页 →", "Next →"], ["与质量助手对话", "Chat with the quality assistant"], ["正在准备分析", "Preparing analysis"], ["查看检查步骤", "View check steps"],
    ["重新分析此数据集 ↻", "Re-analyze this dataset ↻"], ["选择数据集后，追问它的质量、问题与证据…", "Choose a dataset, then ask about its quality, issues and evidence…"],
    ["添加数据集", "Add dataset"], ["选择或上传数据集", "Choose or upload a dataset"], ["发送消息", "Send message"], ["每个数据集保留一段独立对话", "Each dataset keeps an independent chat"],
    ["分数与数量来自实际检查结果；模型疑点需结合定位证据复核。", "Scores and counts come from actual checks; model candidates require locator evidence for review."],
    ["开始新的检查", "Start a new check"], ["添加一个数据集", "Add a dataset"], ["关闭数据集选择窗口", "Close dataset selector"], ["数据来源", "Data source"],
    ["服务器文件夹", "Server folders"], ["上传 ZIP", "Upload ZIP"], ["服务器", "Server"], ["服务器目录路径", "Server directory path"],
    ["刷新当前目录", "Refresh current directory"], ["筛选当前目录…", "Filter current folder…"], ["筛选当前目录文件夹", "Filter current folders"],
    ["筛选数据类型", "Filter data type"], ["全部类型", "All types"], ["名称", "Name"], ["数据类型 / 内容", "Data type / contents"],
    ["正在读取服务器文件夹…", "Loading server folders…"], ["选择一个数据集文件夹", "Choose a dataset folder"], ["浏览服务器目录，找到想要查看的数据包", "Browse the server folders to find a package"],
    ["添加并查看分析", "Add and view analysis"], ["优先读取已有评价报告，可立即查看结论；报告时间与来源会明确显示。无已有报告时自动启动分析。大型数据包无需下载到本机。", "Existing reports are loaded first so results appear immediately; source and time are shown. If no report exists, analysis starts automatically. Large packages stay on the server."],
    ["拖放 ZIP 到这里", "Drop a ZIP here"], ["或点击选择文件", "or click to choose a file"], ["数据文件与标注一起打包 · 一次上传一个数据集", "Package data and labels together · one dataset at a time"], ["取消上传", "Cancel upload"],
    ["上传完成后自动建立会话并分析。大包请使用服务器数据包，避免经浏览器重复传输。", "A session is created and analyzed after upload. For large packages, use a server package to avoid browser transfer."],
    ["样本证据", "Sample evidence"], ["案例详情", "Case details"], ["正在读取样本与定位…", "Loading sample and locator…"], ["原始样本", "Original sample"], ["输入标注", "Input annotation"], ["问题定位", "Issue location"],
    ["查看原始 JSON 标注", "View original JSON label"], ["规则、定位与版本记录", "Rules, locator and version record"], ["逐案分析 · 来自实际证据", "Case analysis · from actual evidence"], ["为什么判定这个问题？", "Why was this issue identified?"],
    ["用自然语言解释这个案例", "Explain this case in natural language"], ["只解释当前样本的已有证据，保留确认与待复核状态。", "Explain only the evidence already recorded for this sample; preserve confirmed and review states."],
    ["让大模型解释此案例 ✳", "Ask the language model to explain ✳"], ["到对话中继续追问 →", "Continue in chat →"], ["关闭案例详情", "Close case details"],
    ["已通过", "Passed"], ["等待分析", "Waiting"], ["正在分析", "Analyzing"], ["分析完成", "Analysis complete"], ["部分完成", "Partially complete"], ["分析失败", "Analysis failed"], ["等待执行", "Waiting to run"], ["已跳过", "Skipped"], ["已取消", "Cancelled"], ["状态待更新", "Status pending"],
    ["还没有数据集对话。添加一个数据包，开始第一次检查。", "No dataset chats yet. Add a package to start the first check."], ["暂时无法读取会话。已有数据不会因刷新页面而删除。", "Sessions cannot be read right now. Existing data is not deleted by refreshing."],
    ["未命名数据集", "Unnamed dataset"], ["待识别", "Unidentified"], ["未分类", "Unclassified"], ["未提供", "Not provided"], ["未知", "Unknown"],
    ["文件缺失", "Missing data file"], ["边界与位置偏差", "Boundary / position deviation"], ["类别标注错误", "Wrong annotation class"], ["起止时刻抖动", "Start/end time jitter"], ["元数据尺寸矛盾", "Metadata size contradiction"], ["漏标", "Missing annotation"], ["虚标", "Spurious annotation"], ["覆盖不足或过度", "Under/over coverage"],
  ];

  const enToZh = Object.fromEntries(pairs);
  const zhToEn = Object.fromEntries(pairs.map(([en, zh]) => [zh, en]));
  const sourceLang = () => document.body?.dataset.localeSource || "en";
  const targetLang = () => localStorage.getItem("spaceqa.language") || "zh";
  const excluded = (node) => {
    const parent = node.parentElement;
    return !parent || parent.closest("script,style,pre,code,[data-no-translate],.message-content,.case-llm-answer,.raw-data") !== null;
  };
  function translateText(value, target, source) {
    if (!value || !value.trim() || target === source) return value;
    const table = source === "en" && target === "zh" ? enToZh : zhToEn;
    const exact = table[value.trim()];
    if (exact !== undefined) return value.replace(value.trim(), exact);
    // Dynamic status/card strings retain numbers and identifiers while system
    // phrases are translated. Longer entries are applied first.
    let out = value;
    for (const key of Object.keys(table).sort((a, b) => b.length - a.length)) {
      if (key.length < 2 || !out.includes(key)) continue;
      out = out.split(key).join(table[key]);
    }
    return out;
  }
  function apply(root = document) {
    const target = targetLang();
    const source = sourceLang();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      if (excluded(node)) continue;
      const current = node.nodeValue || "";
      if (!node.__spaceqaSource) node.__spaceqaSource = current;
      const canonical = node.__spaceqaSource;
      const translated = translateText(canonical, target, source);
      node.__spaceqaLast = translated;
      if (current !== translated) node.nodeValue = translated;
    }
    for (const el of root.querySelectorAll?.("[placeholder],[title],[aria-label]") || []) {
      if (el.closest("[data-no-translate],pre,code")) continue;
      for (const attr of ["placeholder", "title", "aria-label"]) {
        if (!el.hasAttribute(attr)) continue;
        const current = el.getAttribute(attr) || "";
        if (!el.dataset[`locale${attr}`]) el.dataset[`locale${attr}`] = current;
        el.setAttribute(attr, translateText(el.dataset[`locale${attr}`], target, source));
      }
    }
    document.documentElement.lang = target === "en" ? "en" : "zh-CN";
    const button = document.getElementById("language-toggle");
    if (button) { button.textContent = target === "en" ? "中文" : "EN"; button.title = target === "en" ? "切换到中文" : "Switch to English"; }
  }
  function set(lang) {
    localStorage.setItem("spaceqa.language", lang === "en" ? "en" : "zh");
    apply(document);
    document.dispatchEvent(new CustomEvent("spaceqa:language-change", { detail: { lang: targetLang() } }));
  }
  function toggle() { set(targetLang() === "en" ? "zh" : "en"); }
  window.SpaceQALocale = { apply, set, toggle, get: targetLang };
  // Defer scripts run after the body has been parsed. Capture canonical text
  // before app.js/chat.js render their first view so a later toggle can always
  // return to the original language.
  apply(document);
  document.addEventListener("DOMContentLoaded", () => {
    apply(document);
    const button = document.getElementById("language-toggle");
    button?.addEventListener("click", toggle);
    let applying = false;
    const observer = new MutationObserver((records) => {
      if (applying) return;
      applying = true;
      try { for (const record of records) { if (record.type === "childList") for (const node of record.addedNodes) if (node.nodeType === Node.ELEMENT_NODE) apply(node); else if (node.nodeType === Node.TEXT_NODE && !excluded(node)) apply(node.parentElement || document); } } finally { applying = false; }
    });
    observer.observe(document.body, { subtree: true, childList: true });
  });
})();
