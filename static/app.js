const $ = (id) => document.getElementById(id);

const messagesEl = $("messages");
const composer = $("composer");
const input = $("input");
const sendBtn = $("send-btn");
const fileInput = $("file-input");
const uploadStatus = $("upload-status");

let sessionId = localStorage.getItem("session_id");

async function ensureSession(forceNew = false) {
  if (sessionId && !forceNew) return sessionId;
  const res = await fetch("/api/session", { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  localStorage.setItem("session_id", sessionId);
  return sessionId;
}

// ---------- 렌더링 ----------
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderInline(text) {
  return escapeHtml(text)
    .replace(/`([^`\n]+)`/g, '<code class="inline">$1</code>')
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
}

// 코드블록(```)과 일반 텍스트를 나눠 렌더링한다. 스트리밍 중 닫히지 않은 블록도 처리한다.
function renderMarkdown(text) {
  const parts = text.split(/```/);
  return parts
    .map((part, i) => {
      if (i % 2 === 0) return renderInline(part);
      const newline = part.indexOf("\n");
      const code = newline >= 0 ? part.slice(newline + 1) : part;
      return `<div class="code-block"><button class="copy-btn" type="button">복사</button><pre><code>${escapeHtml(code.replace(/\n$/, ""))}</code></pre></div>`;
    })
    .join("");
}

messagesEl.addEventListener("click", async (e) => {
  if (!e.target.classList.contains("copy-btn")) return;
  const code = e.target.parentElement.querySelector("code").textContent;
  await navigator.clipboard.writeText(code);
  e.target.textContent = "복사됨";
  setTimeout(() => (e.target.textContent = "복사"), 1500);
});

function addMessage(role, text = "") {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "user") bubble.textContent = text;
  else bubble.innerHTML = renderMarkdown(text);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { wrap, bubble };
}

// ---------- 채팅 (SSE 스트리밍) ----------
async function sendMessage(message) {
  await ensureSession();
  addMessage("user", message);
  const { wrap, bubble } = addMessage("assistant", "");
  bubble.textContent = "…";

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
  let answer = "";

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
      if (data === "[DONE]") continue;
      const payload = JSON.parse(data);
      if (event === "error") {
        wrap.classList.add("error");
        bubble.textContent = payload.message;
        return;
      }
      answer += payload.delta;
      bubble.innerHTML = renderMarkdown(answer);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }
}

composer.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  sendBtn.disabled = true;
  try {
    await sendMessage(message);
  } catch (err) {
    addMessage("error", `오류: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    composer.requestSubmit();
  }
});

// ---------- 파일 업로드 ----------
fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  await ensureSession();
  $("file-label").textContent = file.name;
  uploadStatus.className = "status";
  uploadStatus.textContent = "업로드 중…";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("file", file);

  const res = await fetch("/api/upload", { method: "POST", body: form });
  const data = await res.json();
  if (!res.ok) {
    uploadStatus.className = "status error";
    uploadStatus.textContent = data.detail || "업로드 실패";
    return;
  }

  uploadStatus.textContent = "업로드 완료";
  $("summary-panel").hidden = false;
  $("summary-meta").innerHTML =
    `<div><strong>${escapeHtml(data.filename)}</strong> · ${data.rows}행 × ${data.columns.length}열</div>` +
    `<div class="cols">${data.columns.map(escapeHtml).join(", ")}</div>`;
  $("summary-text").textContent = data.summary;
  fileInput.value = "";
});

// ---------- 새 대화 ----------
$("reset-btn").addEventListener("click", async () => {
  if (sessionId) await fetch(`/api/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
  await ensureSession(true);
  messagesEl.innerHTML = "";
  addMessage("assistant", "새 대화를 시작합니다. 데이터를 업로드하고 질문해 주세요.");
  $("summary-panel").hidden = true;
  $("file-label").textContent = "CSV / Excel 파일 선택";
  uploadStatus.textContent = "";
});

ensureSession();
