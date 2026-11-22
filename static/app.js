const $ = (id) => document.getElementById(id);

const chatEl = $("chat");
const messagesEl = $("messages");
const composer = $("composer");
const input = $("input");
const sendBtn = $("send-btn");
const fileInput = $("file-input");
const welcomeHtml = $("welcome").outerHTML;

let sessionId = localStorage.getItem("session_id");
let busy = false;

async function ensureSession(forceNew = false) {
  if (sessionId && !forceNew) return sessionId;
  const res = await fetch("/api/session", { method: "POST" });
  const data = await res.json();
  setSession(data.session_id);
  return sessionId;
}

function setSession(id) {
  sessionId = id;
  localStorage.setItem("session_id", id);
}

// ---------- 대화 기록 ----------
async function refreshHistory() {
  const res = await fetch("/api/sessions");
  const sessions = res.ok ? await res.json() : [];
  const nav = $("history");
  if (!sessions.length) {
    nav.innerHTML = '<p class="history-empty">저장된 대화가 없습니다.</p>';
    return;
  }
  nav.innerHTML = sessions
    .map(
      (s) =>
        `<div class="nav-item${s.id === sessionId ? " active" : ""}" data-id="${escapeHtml(s.id)}" role="button" tabindex="0" title="${escapeHtml(s.title)}">` +
        `<img src="/static/icons/list.svg" width="18" height="18" alt="" />` +
        `<span class="title">${escapeHtml(s.title)}</span>` +
        `<button class="delete-btn" type="button" title="대화 삭제" aria-label="대화 삭제">✕</button>` +
        `</div>`
    )
    .join("");
}

$("history").addEventListener("click", async (e) => {
  const item = e.target.closest(".nav-item");
  if (!item || busy) return;
  const id = item.dataset.id;

  if (e.target.closest(".delete-btn")) {
    if (!confirm("이 대화와 업로드한 데이터를 삭제할까요? 삭제하면 되돌릴 수 없습니다.")) return;
    await fetch(`/api/session/${id}`, { method: "DELETE" });
    if (id === sessionId) await startNewChat();
    else await refreshHistory();
    return;
  }
  if (id !== sessionId) await loadSession(id);
});

// 저장된 세션을 불러와 화면을 복원한다. 세션이 없으면 false.
async function loadSession(id) {
  const res = await fetch(`/api/session/${id}`);
  if (!res.ok) return false;
  const session = await res.json();

  setSession(id);
  messagesEl.innerHTML = welcomeHtml;
  clearDataInSidebar();

  for (const m of session.messages) {
    if (m.role === "user") addMessage("user", escapeHtml(m.content));
    else if (m.role === "assistant") addMessage("assistant", m.blocks ? renderBlocks(m.blocks) : renderMarkdown(m.content));
    else if (m.role === "file") {
      const summary = session.data && session.data.filename === m.data.filename ? session.data.summary : "";
      addMessage("assistant", fileCardHtml({ ...m.data, summary }));
    }
  }
  if (session.data) showDataInSidebar(session.data);
  await refreshHistory();
  scrollToBottom();
  return true;
}

async function startNewChat() {
  await ensureSession(true);
  messagesEl.innerHTML = welcomeHtml;
  clearDataInSidebar();
  await refreshHistory();
  input.focus();
}

// ---------- 테마 ----------
$("theme-btn").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
});

// ---------- 렌더링 ----------
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderSpan(text) {
  return escapeHtml(text)
    .replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/^#{1,6} (.+)$/gm, '<strong class="md-heading">$1</strong>');
}

// 마크다운 표(| a | b |)를 HTML 표로 바꾸고, 나머지는 인라인 서식만 적용한다.
function renderInline(text) {
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  const isRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isSep = (l) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());

  while (i < lines.length) {
    if (isRow(lines[i]) && i + 1 < lines.length && isSep(lines[i + 1])) {
      const aligns = cells(lines[i + 1]).map((c) => (c.endsWith(":") ? "right" : "left"));
      const head = cells(lines[i]).map((c, k) => `<th style="text-align:${aligns[k] || "left"}">${renderSpan(c)}</th>`).join("");
      const body = [];
      i += 2;
      while (i < lines.length && isRow(lines[i])) {
        body.push(`<tr>${cells(lines[i]).map((c, k) => `<td style="text-align:${aligns[k] || "left"}">${renderSpan(c)}</td>`).join("")}</tr>`);
        i++;
      }
      out.push(`<div class="md-table"><table><thead><tr>${head}</tr></thead><tbody>${body.join("")}</tbody></table></div>`);
      continue;
    }
    out.push(renderSpan(lines[i]) + (i < lines.length - 1 ? "\n" : ""));
    i++;
  }
  return out.join("");
}

// 코드블록(```)과 일반 텍스트를 나눠 렌더링한다. 스트리밍 중 닫히지 않은 블록도 처리한다.
function renderMarkdown(text) {
  return text
    .split(/```/)
    .map((part, i) => {
      if (i % 2 === 0) return renderInline(part);
      const newline = part.indexOf("\n");
      const lang = newline >= 0 ? part.slice(0, newline).trim() : "";
      const code = newline >= 0 ? part.slice(newline + 1) : part;
      return (
        `<div class="code-block"><div class="code-head"><span>${escapeHtml(lang || "code")}</span>` +
        `<button class="copy-btn" type="button">복사</button></div>` +
        `<pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre></div>`
      );
    })
    .join("");
}

// ---------- 에이전트 실행 과정 (텍스트 + 도구 카드) ----------
const TOOL_LABELS = { query_dataset: "데이터 조회", run_python: "코드 실행", run_shell: "터미널 명령" };

function toolStatus(block) {
  if (block.status === "running") return ["cautionary", "실행 중"];
  const failed =
    block.status === "error" ||
    (block.artifact && block.artifact.ok === false) ||
    (typeof block.output === "string" && block.output.startsWith("[조회 실패]"));
  return failed ? ["negative", "실패"] : ["positive", "완료"];
}

function toolCardHtml(block) {
  const [variant, label] = toolStatus(block);
  const name = TOOL_LABELS[block.name] || block.name;
  const args = block.args || {};
  let input;
  if (block.name === "run_python") {
    input = `<details${block.status === "running" ? " open" : ""}><summary>실행한 코드</summary><pre class="tool-code">${escapeHtml(args.code || "")}</pre></details>`;
  } else if (block.name === "run_shell") {
    // pip install 이 uv pip install 로 바뀐 경우 실제 실행한 명령을 보여준다
    const command = (block.artifact && block.artifact.command) || args.command || "";
    input = `<pre class="tool-code">$ ${escapeHtml(command)}</pre>`;
  } else {
    input = `<div class="tool-args">${escapeHtml(Object.keys(args).length ? JSON.stringify(args, null, 0) : "열 정보 조회")}</div>`;
  }

  let result = "";
  if (block.status !== "running") {
    const art = block.artifact;
    if (art) {
      const text = [art.output, art.error ? `오류: ${art.error}` : ""].filter((t) => t && t.trim()).join("\n");
      if (text) result += `<pre class="tool-output">${escapeHtml(text)}</pre>`;
      for (const src of art.images || []) result += `<img class="tool-image" src="${escapeHtml(src)}" alt="실행 결과 그래프" />`;
    } else if (block.output) {
      result += `<pre class="tool-output">${escapeHtml(block.output)}</pre>`;
    }
  }

  return (
    `<div class="tool-card card">` +
    `<div class="tool-head"><span class="tool-name">${escapeHtml(name)}</span><span class="badge ${variant}">${label}</span></div>` +
    input +
    result +
    `</div>`
  );
}

function renderBlocks(blocks) {
  return blocks
    .map((b) => {
      if (b.type === "text") return `<div class="text-block">${renderMarkdown(b.text)}</div>`;
      if (b.type === "tool") return toolCardHtml(b);
      if (b.type === "error") return `<div class="inline-error">${escapeHtml(b.text)}</div>`;
      return "";
    })
    .join("");
}

function applyEvent(blocks, ev) {
  if (ev.type === "token") {
    const last = blocks[blocks.length - 1];
    if (last && last.type === "text") last.text += ev.text;
    else blocks.push({ type: "text", text: ev.text });
  } else if (ev.type === "tool_start") {
    blocks.push({ type: "tool", id: ev.id, name: ev.name, args: ev.args, status: "running" });
  } else if (ev.type === "tool_end") {
    const block = blocks.find((b) => b.type === "tool" && b.id === ev.id);
    if (block) Object.assign(block, { status: ev.status === "error" ? "error" : "done", output: ev.output, artifact: ev.artifact });
  }
}

function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(role, html) {
  $("welcome")?.remove();
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = role === "user" ? "avatar" : "avatar ai";
  avatar.textContent = role === "user" ? "나" : "AI";
  wrap.appendChild(avatar);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = html;
  wrap.appendChild(bubble);

  messagesEl.appendChild(wrap);
  scrollToBottom();
  return { wrap, bubble };
}

messagesEl.addEventListener("click", async (e) => {
  const suggestion = e.target.closest(".suggestion");
  if (suggestion) {
    input.value = suggestion.querySelector(".text").textContent;
    composer.requestSubmit();
    return;
  }
  if (!e.target.classList.contains("copy-btn")) return;
  const code = e.target.closest(".code-block").querySelector("code").textContent;
  await navigator.clipboard.writeText(code);
  e.target.textContent = "복사됨";
  setTimeout(() => (e.target.textContent = "복사"), 1500);
});

// ---------- 채팅 (SSE 스트리밍) ----------
async function sendMessage(message) {
  await ensureSession();
  addMessage("user", escapeHtml(message));
  const { wrap, bubble } = addMessage("assistant", '<div class="typing"><span></span><span></span><span></span></div>');

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!res.ok) {
    wrap.classList.add("error");
    bubble.textContent = `요청 실패 (${res.status})`;
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const blocks = [];
  const typing = '<div class="typing"><span></span><span></span><span></span></div>';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const raw of events) {
      let event = "message";
      let data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data || data === "[DONE]") continue;
      const payload = JSON.parse(data);
      if (event === "error") {
        if (!blocks.length) {
          wrap.classList.add("error");
          bubble.textContent = payload.message;
        } else {
          blocks.push({ type: "error", text: payload.message });
          bubble.innerHTML = renderBlocks(blocks);
        }
        return;
      }
      applyEvent(blocks, payload);
      const running = blocks.some((b) => b.type === "tool" && b.status === "running");
      bubble.innerHTML = renderBlocks(blocks) + (running ? typing : "");
      scrollToBottom();
    }
  }
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message || busy) return;
  input.value = "";
  autoResize();
  busy = true;
  sendBtn.disabled = true;
  try {
    await sendMessage(message);
  } catch (err) {
    addMessage("error", escapeHtml(`오류: ${err.message}`)).wrap.classList.add("error");
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
    refreshHistory();
  }
});

function autoResize() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}
input.addEventListener("input", autoResize);

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

// ---------- 파일 업로드 ----------
function colBadges(columns) {
  return columns.map((c) => `<span class="badge">${escapeHtml(c)}</span>`).join("");
}

function fileCardHtml(data) {
  return (
    `<div class="file-card card">` +
    `<div class="head"><span class="title">${escapeHtml(data.filename)}</span><span class="badge positive">불러옴</span></div>` +
    `<div class="meta">${data.rows.toLocaleString()}행 × ${data.columns.length}열 · 이제 이 데이터에 대해 질문해 주세요.</div>` +
    `<div class="cols">${colBadges(data.columns)}</div>` +
    (data.summary ? `<details><summary>데이터 요약 보기</summary><pre>${escapeHtml(data.summary)}</pre></details>` : "") +
    `</div>`
  );
}

function showDataInSidebar(data) {
  $("data-empty").hidden = true;
  $("data-card").hidden = false;
  $("data-name").textContent = data.filename;
  $("data-shape").textContent = `${data.rows.toLocaleString()}행 × ${data.columns.length}열`;
  $("data-cols").innerHTML = colBadges(data.columns);
  $("attached").textContent = `📄 ${data.filename}`;
}

function clearDataInSidebar() {
  $("data-empty").hidden = false;
  $("data-card").hidden = true;
  $("attached").textContent = "";
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (!file) return;
  await ensureSession();

  const { wrap, bubble } = addMessage("assistant", `<strong>${escapeHtml(file.name)}</strong> 업로드 중…`);

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "업로드 실패");
    bubble.innerHTML = fileCardHtml(data);
    showDataInSidebar(data);
    refreshHistory();
  } catch (err) {
    wrap.classList.add("error");
    bubble.textContent = `업로드 실패: ${err.message}`;
  }
  scrollToBottom();
});

// ---------- 새 대화 ----------
// 기존 대화는 지우지 않고 대화 기록에 남겨 둔다
$("reset-btn").addEventListener("click", async () => {
  if (busy) return;
  await startNewChat();
});

// ---------- 시작 ----------
(async () => {
  // 마지막으로 보던 대화를 복원하고, 없으면 새 세션을 만든다
  if (!(sessionId && (await loadSession(sessionId)))) {
    await ensureSession(true);
    await refreshHistory();
  }
})();
