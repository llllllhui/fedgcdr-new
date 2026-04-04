const state = {
  data: null,
  selectedGnn: "",
  selectedDomain: "4",
  selectedRunId: "",
};

const gnnSelect = document.getElementById("gnnSelect");
const domainSelect = document.getElementById("domainSelect");
const runSelect = document.getElementById("runSelect");
const loadBtn = document.getElementById("loadBtn");
const metaInfo = document.getElementById("metaInfo");
const statsGrid = document.getElementById("statsGrid");
const tableBody = document.querySelector("#metricsTable tbody");
const emptyState = document.getElementById("emptyState");
const canvas = document.getElementById("metricsChart");
const ctx = canvas.getContext("2d");

async function init() {
  const res = await fetch("./data/results.json");
  state.data = await res.json();

  setupModelOptions();
  setupEvents();
  refreshRunOptions();
  renderCurrentRun();
}

function setupModelOptions() {
  const types = state.data.summary.gnn_types;
  gnnSelect.innerHTML = types.map((t) => `<option value="${t}">${t.toUpperCase()}</option>`).join("");
  state.selectedGnn = types[0] || "";
  gnnSelect.value = state.selectedGnn;
}

function setupEvents() {
  gnnSelect.addEventListener("change", () => {
    state.selectedGnn = gnnSelect.value;
    refreshRunOptions();
  });

  domainSelect.addEventListener("change", () => {
    state.selectedDomain = domainSelect.value;
    refreshRunOptions();
  });

  loadBtn.addEventListener("click", renderCurrentRun);
}

function getRuns() {
  const grouped = state.data.grouped_runs;
  return grouped?.[state.selectedGnn]?.[state.selectedDomain] || [];
}

function refreshRunOptions() {
  const runs = getRuns();
  runSelect.innerHTML = "";

  if (!runs.length) {
    runSelect.innerHTML = `<option value="">无记录</option>`;
    runSelect.title = "无记录";
    state.selectedRunId = "";
    return;
  }

  runSelect.innerHTML = runs
    .map((r) => {
      const time = r.timestamp.replace("T", " ");
      const shortPath = formatRunLabelPath(r.file);
      const fullLabel = `${time} | ${r.file}`;
      return `<option value="${r.id}" title="${fullLabel}">${time} | ${shortPath}</option>`;
    })
    .join("");

  state.selectedRunId = runs[0].id;
  runSelect.value = state.selectedRunId;
  runSelect.title = runs[0].file;

  runSelect.onchange = () => {
    state.selectedRunId = runSelect.value;
    const selected = getRuns().find((item) => item.id === state.selectedRunId);
    runSelect.title = selected ? selected.file : "";
  };
}

function formatRunLabelPath(path) {
  const normalized = String(path || "").replaceAll("\\", "/");
  const parts = normalized.split("/");
  if (parts.length <= 2) {
    return trimMiddle(normalized, 40);
  }
  const short = `${parts[parts.length - 2]}/${parts[parts.length - 1]}`;
  return trimMiddle(short, 44);
}

function trimMiddle(text, maxLen) {
  if (!text || text.length <= maxLen) return text;
  const keep = Math.max(6, Math.floor((maxLen - 3) / 2));
  return `${text.slice(0, keep)}...${text.slice(-keep)}`;
}

function renderCurrentRun() {
  const run = getRuns().find((item) => item.id === state.selectedRunId) || getRuns()[0];

  if (!run) {
    showEmpty();
    return;
  }

  hideEmpty();
  renderMeta(run);
  renderStats(run);
  renderTable(run);
  renderChart(run);
}

function showEmpty() {
  emptyState.classList.remove("hidden");
  metaInfo.innerHTML = "";
  statsGrid.innerHTML = "";
  tableBody.innerHTML = "";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function hideEmpty() {
  emptyState.classList.add("hidden");
}

function renderMeta(run) {
  const m = run.meta;
  const items = [
    ["数据集", m.dataset || "-"],
    ["训练模型", m.model || "-"],
    ["GNN", (m.gnn_type || "-").toUpperCase()],
    ["源域数量", String(m.num_domain || "-")],
    ["Target Domain", String(m.target_domain ?? "-")],
    ["DP / EPS", `${m.dp ? "ON" : "OFF"} / ${m.eps ?? "-"}`],
    ["随机种子", String(m.random_seed ?? "-")],
    ["学习率", `MF: ${formatLr(m.lr_mf)} / GNN: ${formatLr(m.lr_gnn)}`],
  ];

  metaInfo.innerHTML = items
    .map(([k, v]) => `<div class="meta-item"><b>${k}</b><span>${v}</span></div>`)
    .join("");
}

function renderStats(run) {
  const rounds = run.rounds;
  const final = run.final || rounds[rounds.length - 1] || {};

  const bestHr10 = rounds.reduce((acc, cur) => Math.max(acc, cur.hr10 || 0), 0);
  const bestNdcg10 = rounds.reduce((acc, cur) => Math.max(acc, cur.ndcg10 || 0), 0);

  const cards = [
    ["总记录点", String(rounds.length)],
    ["Final HR@5", fmt(final.hr5)],
    ["Final NDCG@5", fmt(final.ndcg5)],
    ["Final HR@10", fmt(final.hr10)],
    ["Final NDCG@10", fmt(final.ndcg10)],
    ["Best HR@10", fmt(bestHr10)],
    ["Best NDCG@10", fmt(bestNdcg10)],
  ];

  statsGrid.innerHTML = cards
    .map(([k, v]) => `<article class="stat-card"><div class="k">${k}</div><div class="v">${v}</div></article>`)
    .join("");
}

function renderTable(run) {
  const latestByDomain = new Map();
  for (const row of run.rounds) {
    const key = `${row.domain}__${row.phase}`;
    const current = latestByDomain.get(key);
    if (!current || row.round > current.round) {
      latestByDomain.set(key, row);
    }
  }

  const list = Array.from(latestByDomain.values()).sort((a, b) => a.domain.localeCompare(b.domain));
  tableBody.innerHTML = list
    .map(
      (r) => `
      <tr>
        <td>${r.domain}</td>
        <td>${r.phase}</td>
        <td>${r.round}</td>
        <td>${fmt(r.hr5)}</td>
        <td>${fmt(r.ndcg5)}</td>
        <td>${fmt(r.hr10)}</td>
        <td>${fmt(r.ndcg10)}</td>
      </tr>
    `
    )
    .join("");
}

function renderChart(run) {
  const curveRows = pickKtAndFineTuningRows(run);
  const rows = curveRows.length ? curveRows : run.rounds;

  const x = rows.map((_, idx) => idx + 1);
  const y1 = rows.map((r) => r.hr10);
  const y2 = rows.map((r) => r.ndcg10);

  drawTwoLineChart(x, y1, y2, "#be4a2f", "#0d6e6e");
}

function pickKtAndFineTuningRows(run) {
  const targetDomain = resolveTargetDomainName(run);
  const targetRows = run.rounds.filter((r) => r.domain === targetDomain);

  const nonFineRows = targetRows.filter((r) => r.phase !== "Fine-tuning");
  const fineRows = targetRows.filter((r) => r.phase === "Fine-tuning");

  const segments = splitByRoundReset(nonFineRows);
  const ktRows = segments[1] || segments[0] || [];

  return [...ktRows, ...fineRows];
}

function resolveTargetDomainName(run) {
  const fineRows = run.rounds.filter((r) => r.phase === "Fine-tuning");
  if (fineRows.length) return fineRows[0].domain;

  const booksRows = run.rounds.filter((r) => String(r.domain).toLowerCase().includes("books"));
  if (booksRows.length) return booksRows[0].domain;

  return run.rounds[0]?.domain || "";
}

function splitByRoundReset(rows) {
  if (!rows.length) return [];

  const segments = [];
  let current = [rows[0]];

  for (let i = 1; i < rows.length; i += 1) {
    const row = rows[i];
    const prev = rows[i - 1];
    if (row.round <= prev.round) {
      segments.push(current);
      current = [row];
    } else {
      current.push(row);
    }
  }

  segments.push(current);
  return segments;
}
function drawTwoLineChart(xValues, y1, y2, color1, color2) {
  const w = canvas.width;
  const h = canvas.height;
  const pad = { top: 24, right: 24, bottom: 36, left: 48 };
  const chartW = w - pad.left - pad.right;
  const chartH = h - pad.top - pad.bottom;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff9ee";
  ctx.fillRect(0, 0, w, h);

  const ys = [...y1, ...y2];
  const yMinRaw = Math.min(...ys);
  const yMaxRaw = Math.max(...ys);
  const yMin = Math.max(0, yMinRaw - 0.03);
  const yMax = Math.min(1, yMaxRaw + 0.03);

  const getX = (_, idx) => pad.left + (idx / Math.max(1, xValues.length - 1)) * chartW;
  const getY = (v) => pad.top + (1 - (v - yMin) / Math.max(1e-6, yMax - yMin)) * chartH;

  ctx.strokeStyle = "rgba(0,0,0,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const y = pad.top + (i / 5) * chartH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();
  }

  drawSeries(y1, color1, getX, getY);
  drawSeries(y2, color2, getX, getY);

  ctx.fillStyle = "#2b2926";
  ctx.font = "12px Barlow";
  ctx.fillText("HR@10", pad.left, 16);
  ctx.fillText("NDCG@10", pad.left + 68, 16);

  ctx.fillStyle = color1;
  ctx.fillRect(pad.left + 44, 8, 14, 4);
  ctx.fillStyle = color2;
  ctx.fillRect(pad.left + 132, 8, 14, 4);
}

function drawSeries(series, color, getX, getY) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  series.forEach((v, i) => {
    const x = getX(v, i);
    const y = getY(v);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function formatLr(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  const n = Number(v);
  if (n !== 0 && Math.abs(n) < 0.001) return n.toExponential(1);
  return String(n);
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return Number(v).toFixed(4);
}

init().catch((e) => {
  console.error(e);
  alert("加载数据失败，请先生成 data/results.json");
});


