const $ = (selector) => document.querySelector(selector);
const boardSvg = $("#board");
const weightSelect = $("#weightSelect");
const sizeSelect = $("#sizeSelect");
const komiInput = $("#komiInput");
const simulationsSelect = $("#simulationsSelect");
let game = null;
let weights = [];
let humanColor = 1;
let suggestions = [];
let busy = false;
let toastTimer;
const svgNS = "http://www.w3.org/2000/svg";
let renderedSize = 0;
let renderedCells = [];
let stoneLayer = null;
let lastLayer = null;
let suggestionLayer = null;
let lastMarker = null;
let renderedLastAction = null;
let renderedLastStone = 0;
let suggestionNodes = new Map();

async function api(path, payload) {
  const response = await fetch(path, payload ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  } : undefined);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}

function notify(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 2600);
}

function setBusy(value) {
  busy = value;
  updateControls();
}

function updateControls() {
  const active = Boolean(game && !game.terminal && !game.resigned);
  const humanTurn = active && game.player === humanColor;
  $("#undoButton").disabled = busy || !game || !game.can_undo;
  $("#passButton").disabled = busy || !humanTurn;
  $("#analyzeButton").disabled = busy || !humanTurn;
  $("#resignButton").disabled = busy || !active;
  $("#newGameButton").disabled = busy || !weights.length;
  boardSvg.classList.toggle("disabled", busy || !humanTurn);
}

function colorName(color) { return color === 1 ? "黑" : "白"; }
function colorWord(color) { return color === 1 ? "黑棋" : "白棋"; }

function addSvgElement(name, attrs, parent) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  parent.append(node);
  return node;
}

function buildBoard(n) {
  const width = 720;
  const start = 44;
  const end = 676;
  const gap = (end - start) / (n - 1);
  boardSvg.setAttribute("viewBox", `0 0 ${width} ${width}`);
  boardSvg.replaceChildren();
  for (let i = 0; i < n; i++) {
    const x = start + i * gap;
    const y = start + i * gap;
    addSvgElement("line", { x1: start, y1: y, x2: end, y2: y, class: "grid-line" }, boardSvg);
    addSvgElement("line", { x1: x, y1: start, x2: x, y2: end, class: "grid-line" }, boardSvg);
    const col = "ABCDEFGHJKLMNOPQRST"[i];
    const label = addSvgElement("text", { x, y: 20, class: "board-label" }, boardSvg);
    label.textContent = col;
    const rowLabel = addSvgElement("text", { x: 20, y, class: "board-label" }, boardSvg);
    rowLabel.textContent = String(n - i);
  }
  const starIndex = n === 19 ? [3, 9, 15] : n === 13 ? [3, 6, 9] : n === 9 ? [2, 4, 6] : n === 3 ? [1] : [];
  for (const row of starIndex) for (const col of starIndex) {
    addSvgElement("circle", { cx: start + col * gap, cy: start + row * gap, r: Math.max(2.3, gap * .045), class: "star-point" }, boardSvg);
  }
  const intersectionLayer = addSvgElement("g", { class: "intersection-layer" }, boardSvg);
  stoneLayer = addSvgElement("g", { class: "stone-layer" }, boardSvg);
  lastLayer = addSvgElement("g", { class: "last-layer" }, boardSvg);
  suggestionLayer = addSvgElement("g", { class: "suggestion-layer" }, boardSvg);
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    const action = row * n + col;
    const x = start + col * gap;
    const y = start + row * gap;
    addSvgElement("circle", { cx: x, cy: y, r: Math.min(19, gap * .47), class: "intersection", "data-action": action }, intersectionLayer);
  }

  lastMarker = addSvgElement("circle", { class: "last-ring", display: "none" }, lastLayer);
  renderedSize = n;
  renderedCells = Array(n * n).fill(0);
  renderedLastAction = null;
  renderedLastStone = 0;
  suggestionNodes = new Map();
}

function renderStones(n, gap) {
  for (let action = 0; action < n * n; action++) {
    const stone = game.cells[action] || 0;
    if (stone === renderedCells[action]) continue;

    const row = Math.floor(action / n);
    const col = action % n;
    const x = 44 + col * gap;
    const y = 44 + row * gap;
    let node = stoneLayer.querySelector(`[data-stone-action="${action}"]`);
    if (!stone) {
      node?.remove();
    } else if (node) {
      node.setAttribute("class", `stone ${stone === 1 ? "black" : "white"}`);
    } else {
      addSvgElement("circle", {
        cx: x, cy: y, r: gap * .43,
        class: `stone ${stone === 1 ? "black" : "white"}`,
        "data-stone-action": action,
      }, stoneLayer);
    }
    renderedCells[action] = stone;
  }
}

function renderLastMove(n, gap) {
  const action = game.last_action;
  const stone = Number.isInteger(action) && action >= 0 && action < n * n ? game.cells[action] || 0 : 0;
  if (action === renderedLastAction && stone === renderedLastStone) return;
  renderedLastAction = action;
  renderedLastStone = stone;
  if (!stone) {
    lastMarker.setAttribute("display", "none");
    return;
  }
  const row = Math.floor(action / n);
  const col = action % n;
  lastMarker.setAttribute("cx", 44 + col * gap);
  lastMarker.setAttribute("cy", 44 + row * gap);
  lastMarker.setAttribute("r", gap * .18);
  lastMarker.removeAttribute("display");
}

function renderSuggestions(n, gap) {
  const nextActions = new Set();
  for (const [index, move] of suggestions.entries()) {
    if (!Number.isInteger(move.action) || move.action < 0 || move.action >= n * n) continue;
    const action = move.action;
    const row = Math.floor(action / n);
    const col = action % n;
    const x = 44 + col * gap;
    const y = 44 + row * gap;
    let entry = suggestionNodes.get(action);
    if (!entry) {
      const circle = addSvgElement("circle", { cx: x, cy: y, r: gap * .22, class: "suggestion-mark" }, suggestionLayer);
      const number = addSvgElement("text", { x, y: y + .4, class: "suggestion-number" }, suggestionLayer);
      entry = { circle, number };
      suggestionNodes.set(action, entry);
    }
    entry.number.textContent = String(index + 1);
    nextActions.add(action);
  }
  for (const [action, entry] of suggestionNodes) {
    if (nextActions.has(action)) continue;
    entry.circle.remove();
    entry.number.remove();
    suggestionNodes.delete(action);
  }
}

function renderBoard() {
  if (!game) return;
  const n = game.size;
  const gap = (676 - 44) / (n - 1);
  if (renderedSize !== n) buildBoard(n);
  renderStones(n, gap);
  renderLastMove(n, gap);
  renderSuggestions(n, gap);
  boardSvg.classList.toggle("disabled", busy || game.player !== humanColor || game.terminal);
  renderStatus();
}

boardSvg.addEventListener("click", (event) => {
  const point = event.target.closest?.(".intersection");
  if (point && boardSvg.contains(point)) makeMove(Number(point.dataset.action));
});

function renderStatus() {
  const turn = $("#turnIndicator");
  const turnText = turn.lastElementChild;
  turn.classList.toggle("white", game.player === -1);
  turn.classList.toggle("over", game.terminal || game.resigned);
  if (game.terminal || game.resigned) {
    turnText.textContent = "对局结束";
  } else {
    turnText.textContent = game.player === humanColor ? "轮到你" : "引擎行棋";
  }
  $("#moveCount").textContent = String(game.move_number);
  $("#budgetValue").textContent = Number(game.simulations).toLocaleString("zh-CN");
  $("#boardCaption").textContent = `${game.size} 路 · ${game.rules} · ${game.move_number ? `${colorName(game.player)}方行棋` : "黑先"}`;
  const humanStone = humanColor === 1 ? "black" : "white";
  const engineStone = humanColor === 1 ? "white" : "black";
  $("#humanPlayer .player-stone").className = `player-stone ${humanStone}`;
  $("#enginePlayer .player-stone").className = `player-stone ${engineStone}`;
  $("#humanPlayer .player-tag").textContent = `执${colorName(humanColor)}`;
  $("#enginePlayer .player-tag").textContent = `执${colorName(-humanColor)}`;
  $("#humanPlayer").classList.toggle("active", !game.terminal && !game.resigned && game.player === humanColor);
  $("#enginePlayer").classList.toggle("active", !game.terminal && !game.resigned && game.player !== humanColor);

  const latest = game.engine_last;
  $("#engineMove").textContent = latest ? latest.coordinate : "尚无落子";
  $("#engineDetail").textContent = latest
    ? `${latest.visits} / ${latest.simulations} 次 · ${latest.seconds.toFixed(2)} 秒`
    : "开始对局后显示引擎搜索数据";

  const result = $("#gameResult");
  result.hidden = !(game.terminal || game.resigned);
  if (game.resigned) result.textContent = `${colorWord(humanColor)}认输 · ${colorWord(game.result.winner)}胜`;
  else if (game.adjudication) {
    const score = game.adjudication.score_black;
    result.textContent = game.adjudication.no_result ? "无结果" : score === 0 ? "和棋" : `${colorWord(score > 0 ? 1 : -1)}胜 ${Math.abs(score).toFixed(1)} 目`;
  }
  updateControls();
}

async function makeMove(action) {
  if (busy || !game || game.terminal || game.player !== humanColor) return;
  suggestions = [];
  $("#analysisResult").hidden = true;
  setBusy(true);
  try {
    game = await api("/api/play", { action });
  } catch (error) { notify(error.message); }
  finally { setBusy(false); renderBoard(); }
}

async function startGame() {
  if (busy || !weights.length) return;
  humanColor = Number(document.querySelector(".segment.active").dataset.color);
  suggestions = [];
  $("#analysisResult").hidden = true;
  setBusy(true);
  try {
    const settings = {
      weight: weightSelect.value,
      size: Number(sizeSelect.value),
      komi: Number(komiInput.value),
      human_color: humanColor,
    };
    if (simulationsSelect.value !== "auto") settings.simulations = Number(simulationsSelect.value);
    game = await api("/api/new", settings);
    $("#weightMeta").textContent = `step ${game.weight.step.toLocaleString("zh-CN")} · ${game.weight.board_size} 路训练 · ${game.weight.rules}`;
    if (game.engine_move) notify(`引擎先行：${game.engine_move.coordinate}`);
  } catch (error) { notify(error.message); }
  finally { setBusy(false); renderBoard(); }
}

async function undo() {
  if (busy || !game || !game.actions.length) return;
  setBusy(true);
  suggestions = [];
  $("#analysisResult").hidden = true;
  try { game = await api("/api/undo", {}); }
  catch (error) { notify(error.message); }
  finally { setBusy(false); renderBoard(); }
}

async function analyze() {
  if (busy) return;
  setBusy(true);
  try {
    const analysis = await api("/api/analyze", {});
    suggestions = analysis.suggestions;
    $("#winrateValue").textContent = `${(analysis.model.winrate * 100).toFixed(1)}%`;
    $("#candidateList").replaceChildren(...analysis.suggestions.map((move, index) => {
      const item = document.createElement("li");
      item.innerHTML = `<span class="candidate-rank">${index + 1}.</span><span class="candidate-coordinate"></span><span class="candidate-visits"></span>`;
      item.querySelector(".candidate-coordinate").textContent = move.coordinate;
      item.querySelector(".candidate-visits").textContent = `${move.visits} 次`;
      return item;
    }));
    $("#analysisResult").hidden = false;
    renderBoard();
  } catch (error) { notify(error.message); }
  finally { setBusy(false); }
}

async function init() {
  try {
    const data = await api("/api/weights");
    weights = data.weights;
    weightSelect.replaceChildren(...weights.map((weight) => {
      const option = document.createElement("option");
      option.value = weight.id;
      option.textContent = `${weight.label}${weight.id === weights[0].id ? " · 默认" : ""}`;
      return option;
    }));
    weightSelect.disabled = false;
    $("#newGameButton").disabled = false;
    $("#topStatus").textContent = `${weights.length} 个可用权重`;
    game = await api("/api/state");
    sizeSelect.value = [9, 13, 19].includes(game.size) ? String(game.size) : "9";
    komiInput.value = String(game.komi);
    simulationsSelect.value = "auto";
    $("#weightMeta").textContent = `step ${game.weight.step.toLocaleString("zh-CN")} · ${game.weight.board_size} 路训练 · ${game.weight.rules}`;
    renderBoard();
  } catch (error) {
    $("#topStatus").textContent = "服务不可用";
    notify(error.message);
  }
}

document.querySelectorAll(".segment").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".segment").forEach((item) => item.classList.toggle("active", item === button));
}));
weightSelect.addEventListener("change", () => {
  const weight = weights.find((item) => item.id === weightSelect.value);
  if (weight) $("#weightMeta").textContent = `${weight.role} · 新对局时加载`;
});
$("#newGameButton").addEventListener("click", startGame);
$("#undoButton").addEventListener("click", undo);
$("#passButton").addEventListener("click", () => makeMove(game.size * game.size));
$("#analyzeButton").addEventListener("click", analyze);
$("#sgfButton").addEventListener("click", () => { window.location.href = "/api/sgf"; });
$("#resignButton").addEventListener("click", async () => {
  if (!window.confirm("确认认输并结束本局？")) return;
  setBusy(true);
  try { game = await api("/api/resign", {}); }
  catch (error) { notify(error.message); }
  finally { setBusy(false); renderBoard(); }
});
init();
