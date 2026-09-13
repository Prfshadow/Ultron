/* ============================================================
   Ultron - frontend controller (vanilla JS, no frameworks)
   Streams SSE responses, renders markdown, manages UI state.
   ============================================================ */

"use strict";

// ---------- state ----------
const state = {
  chatId: null,
  providers: [],              // fetched from settings
  settings: {},
  attachments: [],            // pending uploads for [next message]
  messages: [],               // rendered messages for current chat
  streaming: false,
  abortController: null,
  editingMemoryId: null,
  currentProvider: "auto",
  theme: "dark",
  voiceSend: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// ---------- tiny helpers ----------
async function api(url, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && typeof opts.body !== "string" && !(opts.body instanceof FormData)) {
    opts.body = JSON.stringify(opts.body);
    opts.headers["Content-Type"] = "application/json";
  }
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try { const j = await res.json(); msg = j.message || j.error || msg; } catch (e) {}
    throw new Error(msg);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("json") ? res.json() : res.text();
}

let toastTimer = null;
function toast(msg, type = "") {
  const el = $("#toast");
  el.textContent = msg;
  el.className = "toast show " + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.classList.remove("show"); }, 2600);
}

// ---------- markdown ----------
marked.setOptions({
  breaks: true,
  highlight(code, lang) {
    try {
      if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang }).value;
      return hljs.highlightAuto(code).value;
    } catch (e) { return code; }
  },
});

function renderMarkdown(text) {
  if (!window.marked) return escapeHtml(text);
  const raw = marked.parse(text || "");
  return DOMPurify.sanitize(raw);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// highlight code blocks after render
function applyHighlight(container) {
  container.querySelectorAll("pre code").forEach((block) => {
    hljs.highlightElement(block);
  });
}

const icons = {
  image: "img", doc: "doc", txt: "txt", pdf: "pdf",
};

// Ultron avatar — a glowing red orb (animated via CSS).
const ULTRON_AVATAR = `
  <svg viewBox="0 0 40 40" class="ultron-avatar" aria-hidden="true" focusable="false">
    <defs>
      <radialGradient id="orb-core" cx=".38" cy=".3" r="1">
        <stop offset="0" stop-color="#ff8a3c"/>
        <stop offset=".4" stop-color="#ff2d2d"/>
        <stop offset=".85" stop-color="#7a0000"/>
        <stop offset="1" stop-color="#3d0000"/>
      </radialGradient>
      <radialGradient id="orb-halo" cx=".5" cy=".5" r=".5">
        <stop offset="0" stop-color="#ff2d2d" stop-opacity=".55"/>
        <stop offset=".6" stop-color="#ff2d2d" stop-opacity=".15"/>
        <stop offset="1" stop-color="#ff2d2d" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <circle class="ua-halo" cx="20" cy="20" r="18" fill="url(#orb-halo)"/>
    <circle class="ua-ring" cx="20" cy="20" r="15" fill="none" stroke="#ff2d2d" stroke-width="1.6" stroke-dasharray="3 5" stroke-linecap="round" opacity=".9"/>
    <circle class="ua-orb" cx="20" cy="20" r="8.5" fill="url(#orb-core)"/>
    <ellipse cx="16.5" cy="16.5" rx="3.4" ry="2.2" fill="#fff" opacity=".55" transform="rotate(-30 16.5 16.5)"/>
  </svg>`;

// User avatar — simple person silhouette matching the theme.
const USER_AVATAR = `
  <svg viewBox="0 0 40 40" class="user-avatar" aria-hidden="true" focusable="false">
    <defs>
      <linearGradient id="uua-grad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#5b6170"/>
        <stop offset="1" stop-color="#1c1f28"/>
      </linearGradient>
    </defs>
    <circle cx="20" cy="15" r="6.5" fill="url(#uua-grad)"/>
    <path d="M7 35 A13 13 0 0 1 33 35 Z" fill="url(#uua-grad)"/>
  </svg>`;

// ---------- theme ----------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const icon = $("#themeIcon");
  if (icon) icon.innerHTML = theme === "dark" ? '<use href="#i-sun"/>' : '<use href="#i-moon"/>';
}

function setTheme(theme) {
  state.theme = theme;
  applyTheme(theme);
  localStorage.setItem("ultron-theme", theme);
  const cb = $("#setThemeDark");
  if (cb) cb.checked = theme === "dark";
}

function toggleTheme() {
  setTheme(state.theme === "dark" ? "light" : "dark");
}

// ---------- settings ----------
async function loadSettings() {
  const s = await api("/api/settings");
  state.settings = s;
  state.currentProvider = ["auto", "groq", "gemini"].includes(s.provider) ? s.provider : "auto";
  $("#providerSelect").value = state.currentProvider;
  fillSettingsForm(s);
}

function fillSettingsForm(s) {
  $("#setProvider").value = s.provider || "auto";
  $("#setCyfutureModel").value = s.cyfuture_model || "";
  $("#setCyfutureVisionModel").value = s.cyfuture_vision_model || "";
  $("#setCyfutureBaseUrl").value = s.cyfuture_base_url || "";
  $("#setGroqModel").value = s.groq_model || "";
  $("#setGroqVisionModel").value = s.groq_vision_model || "";
  $("#setGeminiModel").value = s.gemini_model || "";
  $("#setGeminiVisionModel").value = s.gemini_vision_model || "";

  $("#setTemperature").value = s.temperature ?? 0.7;
  $("#tempValue").textContent = s.temperature ?? 0.7;
  $("#setMaxTokens").value = s.max_tokens ?? 1024;
  $("#setRag").checked = isTruthy(s.rag_enabled);
  $("#setMemory").checked = isTruthy(s.memory_enabled);
  $("#setMemoryAuto").checked = isTruthy(s.memory_auto);
  $("#setTools").checked = isTruthy(s.tools_enabled);
  $("#setProviderFallback").checked = isTruthy(s.provider_fallback);
  $("#setDefaultCity").value = s.default_city || "";
  $("#setWebResults").value = s.web_results ?? 4;
  $("#setKbNamespace").value = s.kb_namespace || "";
  $("#setThemeDark").checked = document.documentElement.getAttribute("data-theme") === "dark";
}

function isTruthy(v) { return v === true || v === "true" || v === "1"; }

function readSettingsForm() {
  return {
    provider: $("#setProvider").value,
    cyfuture_model: $("#setCyfutureModel").value.trim(),
    cyfuture_vision_model: $("#setCyfutureVisionModel").value.trim(),
    cyfuture_base_url: $("#setCyfutureBaseUrl").value.trim(),
    groq_model: $("#setGroqModel").value.trim(),
    groq_vision_model: $("#setGroqVisionModel").value.trim(),
    gemini_model: $("#setGeminiModel").value.trim(),
    gemini_vision_model: $("#setGeminiVisionModel").value.trim(),
    temperature: parseFloat($("#setTemperature").value),
    max_tokens: parseInt($("#setMaxTokens").value, 10),
    rag_enabled: $("#setRag").checked,
    memory_enabled: $("#setMemory").checked,
    memory_auto: $("#setMemoryAuto").checked,
    tools_enabled: $("#setTools").checked,
    provider_fallback: $("#setProviderFallback").checked,
    default_city: $("#setDefaultCity").value.trim(),
    web_results: parseInt($("#setWebResults").value, 10) || 4,
    kb_namespace: $("#setKbNamespace").value.trim(),
  };
}

// ---------- chats ----------
async function listChats() {
  const chats = await api("/api/chats");
  renderChatList(chats, state.chatId);
}

function renderChatList(chats, activeId) {
  const list = $("#chatList");
  if (!chats.length) {
    list.innerHTML = '<div class="sidebar-empty">No chats yet.<br>Start a new conversation.</div>';
    return;
  }
  list.innerHTML = chats.map((c) => `
    <div class="chat-item ${c.id === activeId ? "active" : ""}" data-id="${c.id}">
      <span class="dot"></span>
      <span class="title" title="${escapeHtml(c.preview || c.title)}">${escapeHtml(c.title)}</span>
      <span class="row-actions">
        <button class="rename" title="Rename"><svg><use href="#i-edit"/></svg></button>
        <button class="del" title="Delete"><svg><use href="#i-trash"/></svg></button>
      </span>
    </div>`).join("");

  list.querySelectorAll(".chat-item").forEach((item) => {
    item.addEventListener("click", (e) => {
      if (e.target.closest(".rename")) return renameChat(item);
      if (e.target.closest(".del")) return confirmDeleteChat(item);
      openChat(parseInt(item.dataset.id, 10));
    });
    item.querySelector(".del").addEventListener("click", (e) => {
      e.stopPropagation();
      confirmDeleteChat(item);
    });
    item.querySelector(".rename").addEventListener("click", (e) => {
      e.stopPropagation();
      renameChat(item);
    });
  });
}

async function renameChat(item) {
  const id = parseInt(item.dataset.id, 10);
  const title = (prompt("Rename chat:", item.querySelector(".title").textContent) || "").trim();
  if (!title) return;
  await api(`/api/chats/${id}`, { method: "PATCH", body: { title } });
  await listChats();
  if (id === state.chatId) $("#chatTitle").textContent = title;
}

function confirmDeleteChat(item) {
  const id = parseInt(item.dataset.id, 10);
  $("#confirmTitle").textContent = "Delete chat?";
  $("#confirmText").textContent = "This will permanently remove all messages in this chat.";
  openModal("#confirmModal");
  $("#confirmOk").onclick = async () => {
    await api(`/api/chats/${id}`, { method: "DELETE" });
    closeModals();
    if (id === state.chatId) { state.chatId = null; state.messages = []; renderConversation([]); }
    toast("Chat deleted");
    await listChats();
  };
}

async function openChat(id) {
  if (state.streaming) stopGeneration();
  state.attachments = [];
  renderAttachmentChips();
  const data = await api(`/api/chats/${id}`);
  state.chatId = id;
  $(".chat-item.active")?.classList.remove("active");
  $$(".chat-item").forEach((el) => el.classList.toggle("active", el.dataset.id == id));
  $("#chatTitle").textContent = data.chat.title;
  renderConversation(data.messages);
  await listChats();
}

async function newChat() {
  if (state.streaming) stopGeneration();
  state.attachments = [];
  renderAttachmentChips();
  state.chatId = null;
  state.messages = [];
  $("#chatTitle").textContent = "New Chat";
  renderConversation([]);
  setEmpty(true);
  await listChats();
}

// ---------- messages / conversation ----------
const $messages = () => $("#messages");

function setEmpty(show) {
  $("#emptyState").classList.toggle("hidden", !show);
  $("#conversation").classList.toggle("hidden", show);
}

function renderConversation(messages) {
  state.messages = messages;
  $("#conversation").innerHTML = "";
  if (!messages.length) { setEmpty(true); return; }
  setEmpty(false);
  messages.forEach((m) => appendMessageEl(m, false));
  scrollBottom(true);
}

function appendMessageEl(m, animate) {
  const conv = $("#conversation");
  setEmpty(false);

  if (m.role === "assistant") {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.dataset.mid = m.id;
    const files = safeFiles(m.files);
    wrap.innerHTML = `
      <div class="avatar">${ULTRON_AVATAR}</div>
      <div class="bubble">
        <div class="md"></div>
        <div class="msg-actions">
          <button class="act-copy" title="Copy"><svg><use href="#i-copy"/></svg> Copy</button>
          <button class="act-speak" title="Read aloud"><svg><use href="#i-speak"/></svg> Speak</button>
          <button class="act-regen" title="Regenerate"><svg><use href="#i-refresh"/></svg> Regenerate</button>
        </div>
        ${files.filter(f => f && f.url).map(f => `<div class="md">![${escapeHtml(f.name)}](${f.url})</div>`).join("")}
        <div class="model-badge">${escapeHtml(m.model || "")}</div>
        ${renderSourcesHTML(safeSources(m.sources))}
        ${m.status === "stopped" ? '<div class="stopped-tag">stopped</div>' : ""}
      </div>`;
    wrap.querySelector(".md").innerHTML = renderMarkdown(m.content);
    applyHighlight(wrap);
    wrap.querySelector(".act-copy").onclick = () => copyText(m.content);
    wrap.querySelector(".act-speak").onclick = (e) => speakMessage(e.currentTarget, m.content);
    wrap.querySelector(".act-regen").onclick = () => regenerate();
    conv.appendChild(wrap);
    if (animate) wrap.style.animation = "msgIn 0.35s ease both";
    return wrap;
  }

  // user message
  const files = safeFiles(m.files);
  const wrap = document.createElement("div");
  wrap.className = "msg user";
  wrap.dataset.mid = m.id;
  wrap.innerHTML = `
    <div class="avatar">${USER_AVATAR}</div>
    <div class="bubble">
      <div class="text">${escapeHtml(m.content)}</div>
      ${renderFileChipsHTML(files)}
      <div class="msg-actions user-actions">
        <button class="act-rewrite" title="Edit"><svg><use href="#i-rewrite"/></svg> Edit</button>
      </div>
    </div>`;
  conv.appendChild(wrap);
  wrap.querySelector(".act-rewrite").onclick = () => editMessage(wrap, m.id, m.content);
  if (animate) wrap.style.animation = "msgIn 0.3s ease both";
  return wrap;
}

function safeFiles(files) {
  if (!files) return [];
  try { return Array.isArray(files) ? files : JSON.parse(files); }
  catch (e) { return []; }
}

function safeSources(src) {
  if (!src) return [];
  try { return Array.isArray(src) ? src : JSON.parse(src); }
  catch (e) { return []; }
}

function renderSourcesHTML(sources) {
  if (!sources || !sources.length) return "";
  const chips = sources.map((s) => {
    const label = s.pages ? `${s.filename} · p. ${s.pages}` : s.filename;
    const open = s.path ? ` onclick="window.open('/uploads/${encodeURIComponent(s.path)}')" title="Open ${escapeHtml(s.filename)}"` : "";
    return `<button class="src-chip"${open}>${escapeHtml(label)}</button>`;
  }).join("");
  return `<div class="sources"><span class="src-label">Sources:</span>${chips}</div>`;
}

function renderFileChipsHTML(files) {
  if (!files.length) return "";
  const chips = files.map((f) => {
    const inner = f.kind === "image" && f.url ? `<img src="${f.url}" alt="">` : "🞩";
    return `<span class="attach-chip"><span class="icon">${inner}</span>${escapeHtml(f.name)}</span>`;
  }).join("");
  return `<div class="attach-row">${chips}</div>`;
}

function scrollBottom(force = false) {
  const el = $messages();
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 160;
  if (force || nearBottom) el.scrollTop = el.scrollHeight;
}

// ---------- sending / streaming ----------
const input = $("#input");
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}

function sendMessage() {
  if (state.streaming) return;
  const text = input.value.trim();
  if (!text && !state.attachments.length) return;

  // Prevent sending while uploads are in progress
  if (state.attachments.some(a => a.pending)) {
    toast("Please wait for file uploads to complete", "error");
    return;
  }

  stopSpeaking();

  const files = state.attachments.map((a) => ({ id: a.id, name: a.name, kind: a.kind, url: a.url }));
  state.attachments = [];
  renderAttachmentChips();
  input.value = "";
  autoGrow();
  const speakReply = state.voiceSend ? speakVoiceReply : undefined;
  state.voiceSend = false;
  stream({ message: text, files, chat_id: state.chatId, regenerate: false }, true, speakReply);
}

function regenerate() {
  if (state.streaming) return;
  stopSpeaking();
  const userMsg = [...state.messages].reverse().find((m) => m.role === "user");
  if (!userMsg) return;
  const files = safeFiles(userMsg.files).map((f) => ({ id: f.id, name: f.name, kind: f.kind, url: f.url }));
  // trim trailing assistant messages
  let idx = state.messages.lastIndexOf(userMsg);
  state.messages = state.messages.slice(0, idx + 1);
  stream({ message: userMsg.content, files, chat_id: state.chatId, regenerate: true }, false);
}

let _editMsgId = null;
let _editOriginal = "";

function editMessage(msgEl, msgId, originalText) {
  if (state.streaming) return;
  _editMsgId = msgId;
  _editOriginal = originalText;

  const textEl = msgEl.querySelector(".text");
  const actionsEl = msgEl.querySelector(".user-actions");

  // Replace text with textarea
  textEl.innerHTML = `<textarea class="edit-input">${escapeHtml(originalText)}</textarea>`;
  const textarea = textEl.querySelector("textarea");
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);

  // Replace rewrite button with save/cancel
  actionsEl.innerHTML = `
    <button class="act-edit-save" title="Save"><svg><use href="#i-check"/></svg> Save</button>
    <button class="act-edit-cancel" title="Cancel"><svg><use href="#i-close"/></svg> Cancel</button>`;

  actionsEl.querySelector(".act-edit-save").onclick = () => saveEdit(msgEl, msgId);
  actionsEl.querySelector(".act-edit-cancel").onclick = () => cancelEdit(msgEl, originalText);

  // Save on Enter, cancel on Escape
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); saveEdit(msgEl, msgId); }
    if (e.key === "Escape") cancelEdit(msgEl, originalText);
  });
}

function saveEdit(msgEl, msgId) {
  const textarea = msgEl.querySelector(".edit-input");
  const newText = textarea.value.trim();
  if (!newText || newText === _editOriginal) {
    cancelEdit(msgEl, _editOriginal);
    return;
  }

  // Find the message in state and update it
  const msg = state.messages.find((m) => m.id === msgId);
  if (msg) msg.content = newText;

  // Delete this message and all following messages from server, then re-send
  const idx = state.messages.indexOf(msg);
  state.messages = state.messages.slice(0, idx + 1);

  // Find files from the original message
  const files = safeFiles(msg.files).map((f) => ({ id: f.id, name: f.name, kind: f.kind, url: f.url }));

  // Restore the message display
  const textEl = msgEl.querySelector(".text");
  textEl.textContent = newText;
  const actionsEl = msgEl.querySelector(".user-actions");
  actionsEl.innerHTML = `<button class="act-rewrite" title="Edit"><svg><use href="#i-rewrite"/></svg> Edit</button>`;
  actionsEl.querySelector(".act-rewrite").onclick = () => editMessage(msgEl, msgId, newText);

  stream({ message: newText, files, chat_id: state.chatId, rewrite_from: msgId }, false);
}

function cancelEdit(msgEl, originalText) {
  const textEl = msgEl.querySelector(".text");
  textEl.textContent = originalText;
  const actionsEl = msgEl.querySelector(".user-actions");
  actionsEl.innerHTML = `<button class="act-rewrite" title="Edit"><svg><use href="#i-rewrite"/></svg> Edit</button>`;
  actionsEl.querySelector(".act-rewrite").onclick = () => editMessage(msgEl, _editMsgId, originalText);
}

async function stream(body, appendUser, onDone) {
  if (!body.chat_id) {
    const created = await api("/api/chats", { method: "POST" });
    body.chat_id = created.id;
    state.chatId = created.id;
    const first = (body.message || "New Chat").split(/\s+/).slice(0, 6).join(" ");
    $("#chatTitle").textContent = first || "New Chat";
    await listChats();
  }

  // create an assistant placeholder with typing dots
  if (appendUser) {
    appendMessageEl({ role: "user", content: body.message || "", files: body.files.map(f => ({ name: f.name, kind: f.kind, url: f.url })) }, true);
  }
  const thinking = appendThinkingBubble();
  state.streaming = true;
  $("#sendBtn").classList.add("hidden");
  $("#stopBtn").classList.remove("hidden");
  $("#modelDot").classList.add("pulse");
  scrollBottom(true);

  state.abortController = new AbortController();
  let acc = "";
  const emptyStateSnapshot = { chunks: [] };

  try {
    // reuse the thinking bubble as the live assistant bubble
    thinking.innerHTML = htmlAssistantShell();
    const bodyEl = thinking.querySelector(".md");
    const badgeEl = thinking.querySelector(".model-badge");
    const sourcesEl = thinking.querySelector(".sources");

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal: state.abortController.signal,
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const raw = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        raw.split("\n").forEach((line) => {
          if (line.startsWith("data:")) {
            processEvent(JSON.parse(line.slice(5).trim()), { bodyEl, badgeEl, sourcesEl, setAcc: (s) => acc = s, getAcc: () => acc });
          }
        });
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      console.error(err);
      thinking.querySelector(".md").innerHTML = renderMarkdown("*Connection error. Please try again.*");
      toast("Connection error", "error");
    }
  } finally {
    const doneText = acc;
    finalizeMessage(thinking, true);
    if (onDone && doneText) onDone(doneText);
  }
}

function processEvent(ev, ref) {
  switch (ev.type) {
    case "token":
      ref.setAcc(ref.getAcc() + ev.content);
      // Throttle expensive markdown re-render to ~every 120ms so fast
      // streams don't freeze the UI; final text is rendered on done.
      const now = Date.now();
      if (!ref._lastRender || now - ref._lastRender > 120) {
        ref._lastRender = now;
        ref.bodyEl.innerHTML = renderMarkdown(ref.getAcc());
        applyHighlight(ref.bodyEl.closest(".msg"));
        scrollBottom(false);
      } else {
        ref._pendingRender = true;
      }
      break;
    case "meta":
      ref.badgeEl.textContent = `${ev.provider} · ${ev.model}`;
      state.currentProvider = ev.provider;
      if (ref._pendingRender) {
        ref._pendingRender = false;
        ref._lastRender = Date.now();
        ref.bodyEl.innerHTML = renderMarkdown(ref.getAcc());
        applyHighlight(ref.bodyEl.closest(".msg"));
      }
      break;
    case "fallback":
      ref.badgeEl.textContent = `fallback → ${ev.to}`;
      break;
    case "tool":
      ref.badgeEl.textContent = ev.message || "Using tool...";
      if (ev.kind === "imagegen" && ev.image_url) {
        const img = document.createElement("img");
        img.src = ev.image_url;
        img.alt = ev.prompt || "Generated image";
        img.style.maxWidth = "360px";
        img.style.maxHeight = "360px";
        img.style.width = "auto";
        img.style.height = "auto";
        img.style.borderRadius = "12px";
        img.style.marginTop = "8px";
        img.style.boxShadow = "0 4px 24px rgba(0,0,0,0.3)";
        img.style.display = "block";
        img.style.objectFit = "contain";
        img.loading = "lazy";
        // Add to accumulated text for persistence (as markdown)
        const md = "\n\n![" + (ev.prompt || "Generated image") + "](" + ev.image_url + ")";
        ref.setAcc(ref.getAcc() + md);
        // Also append actual img element that persists through re-renders
        ref.bodyEl.appendChild(img);
      }
      scrollBottom(false);
      break;
    case "error":
      toast(ev.message || "All AI providers failed", "error");
      ref.bodyEl.innerHTML = "";
      break;
    case "sources":
      if (ref.sourcesEl) ref.sourcesEl.innerHTML = renderSourcesHTML(ev.sources || []);
      break;
  }
}

function htmlAssistantShell() {
  return `
    <div class="avatar">${ULTRON_AVATAR}</div>
    <div class="bubble">
      <div class="md"><div class="typing"><span></span><span></span><span></span></div></div>
      <div class="msg-actions">
        <button class="act-copy" title="Copy"><svg><use href="#i-copy"/></svg> Copy</button>
        <button class="act-speak" title="Read aloud"><svg><use href="#i-speak"/></svg> Speak</button>
        <button class="act-regen" title="Regenerate"><svg><use href="#i-refresh"/></svg> Regenerate</button>
      </div>
      <div class="model-badge"></div>
      <div class="sources"></div>
    </div>`;
}

let thinkingMsgEl = null;
function appendThinkingBubble() {
  const div = document.createElement("div");
  div.className = "msg assistant streaming";
  div.innerHTML = htmlAssistantShell();
  $("#conversation").appendChild(div);
  return div;
}

function finalizeMessage(msgEl) {
  const text = msgEl.querySelector(".md").innerHTML;
  const plain = msgEl.querySelector(".md").textContent || "";
  msgEl.querySelector(".act-copy").onclick = () => copyText(plain);
  msgEl.querySelector(".act-speak").onclick = (e) => speakMessage(e.currentTarget, plain);
  msgEl.querySelector(".act-regen").onclick = () => regenerate();
  state.streaming = false;
  $("#stopBtn").classList.add("hidden");
  $("#sendBtn").classList.remove("hidden");
  $("#modelDot").classList.remove("pulse");
  scrollBottom(true);
  // reload history so the server-side saved message is reflected
  if (state.chatId) {
    api(`/api/chats/${state.chatId}`)
      .then((data) => {
        state.messages = data.messages;
        renderConversation(data.messages);
      })
      .catch((err) => {
        console.error("Failed to reload chat:", err);
        toast("Chat saved but UI refresh failed", "error");
      });
  }
  setEmpty(false);
}

function stopGeneration() {
  if (state.abortController) state.abortController.abort();
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text || "");
    toast("Copied");
  } catch (e) { toast("Copy failed", "error"); }
}

// ---------- read-aloud (text-to-speech) ----------
let speakState = null;
let _ttsVoices = null;

function speechAvailable() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

function loadVoices() {
  if (!speechAvailable()) return;
  _ttsVoices = speechSynthesis.getVoices();
}

function savedVoiceName() {
  try { return localStorage.getItem("ultron-voice") || ""; } catch (e) { return ""; }
}

function pickVoice() {
  loadVoices();
  const voices = _ttsVoices || [];
  if (!voices.length) return null;

  // 1) the voice the user explicitly chose in Settings
  const saved = savedVoiceName();
  if (saved) {
    const chosen = voices.find((v) => v.name === saved) || voices.find((v) => v.name.toLowerCase() === saved.toLowerCase());
    if (chosen) return chosen;
  }

  // 2) deepest, most authoritative male voices first (Ultron-like: Spader's deep baritone)
  const male = ["Microsoft Christopher", "Microsoft Steffan", "Microsoft Guy", "Microsoft Eric", "Microsoft Roger", "Microsoft George", "Google UK English Male", "Microsoft James", "Microsoft David", "Microsoft Mark", "Microsoft Ryan", "Microsoft Noah", "Daniel", "Microsoft Ricardo"];
  for (const name of male) {
    const v = voices.find((x) => x.name.toLowerCase().includes(name.toLowerCase()));
    if (v) return v;
  }

  // 3) any English voice, defaulting to default
  return voices.find((v) => /^en/i.test(v.lang)) || voices[0] || null;
}

async function populateVoiceSelect() {
  if (!$("#setVoice")) return;
  const saved = savedVoiceName();
  const prev = $("#setVoice").value;
  let opts = ['<option value="">Auto (Ultron — deep, menacing)</option>'];
  try {
    const j = await api("/api/tts/voices");
    state.edgeTts = !!j.engine;
    if (j.voices && j.voices.length) {
      for (const v of j.voices) opts.push(`<option value="${escapeHtml(v.id)}">${escapeHtml(v.name)}</option>`);
    }
  } catch (e) {
    state.edgeTts = false;
  }
  $("#setVoice").innerHTML = opts.join("");
  const want = prev || saved || "";
  if (want) $("#setVoice").value = Array.from($("#setVoice").options).some((o) => o.value === want) ? want : "";
}

function mdToPlain(md) {
  let t = (md || "")
    .replace(/```([^\n]*)\n?([\s\S]*?)```/g, (m, lang, code) => {
      const spoken = code
        .replace(/\s*=>\s*/g, " arrow ")
        .replace(/\s*===\s*/g, " equals ")
        .replace(/\s*==\s*/g, " equals ")
        .replace(/\s*&&\s*/g, " and ")
        .replace(/\s*\|\|\s*/g, " or ")
        .replace(/\s*=\s*/g, " equals ")
        .replace(/;/g, ". ")
        .replace(/\n+/g, ". ")
        .replace(/\s+/g, " ")
        .trim();
      return spoken ? " Code: " + spoken + " End of code. " : " ";
    })
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+[.)]\s*/gm, "")
    .replace(/\|/g, "")
    .replace(/^\s*={3,}\s*$/gm, "")
    .replace(/^\s*[-*_]\s*[-*_\s]\s*$/gm, "")
    .replace(/^>+\s*/gm, "")
    .replace(/[*_~#]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return t;
}

function chunkSpeech(text) {
  const chunks = [];
  const max = 400;
  let cur = "";
  for (const word of text.split(" ")) {
    if (cur && cur.length + word.length + 1 > max) {
      chunks.push(cur);
      cur = word;
    } else {
      cur = cur ? cur + " " + word : word;
    }
  }
  if (cur) chunks.push(cur);
  return chunks;
}

function setSpeakLabel(btn, label) {
  if (!btn) return;
  const text = Array.from(btn.childNodes).find((n) => n.nodeType === 3 && n.textContent.trim());
  if (text) text.textContent = " " + label;
}

function clearSpeakState() {
  if (speakState) {
    try {
      speakState.btn.classList.remove("on");
      speakState.btn.classList.remove("paused");
      setSpeakLabel(speakState.btn, "Speak");
    } catch (e) {}
  }
  speakState = null;
}

function pauseSpeech() {
  if (!speakState) return;
  speakState.paused = true;
  if (speakState.mode === "edge") {
    try { getTtsAudio().pause(); } catch (e) {}
  } else if (speechAvailable()) {
    speechSynthesis.pause();
  }
  try {
    speakState.btn.classList.remove("on");
    speakState.btn.classList.add("paused");
    setSpeakLabel(speakState.btn, "Resume");
  } catch (e) {}
}

function resumeSpeech() {
  if (!speakState) return;
  speakState.paused = false;
  if (speakState.mode === "edge") {
    try { getTtsAudio().play().catch(() => {}); } catch (e) {}
  } else if (speechAvailable()) {
    speechSynthesis.resume();
    if (!speechSynthesis.speaking && speakState.next) setTimeout(() => { if (speakState && !speakState.paused) speakState.next(); }, 50);
  }
  try {
    speakState.btn.classList.remove("paused");
    speakState.btn.classList.add("on");
    setSpeakLabel(speakState.btn, "Pause");
  } catch (e) {}
}

function stopSpeaking() {
  if (speechAvailable()) speechSynthesis.cancel();
  if (edgeAbort) { try { edgeAbort.abort(); } catch (e) {} edgeAbort = null; }
  if (ttsAudioEl) { try { ttsAudioEl.pause(); } catch (e) {} ttsAudioEl.removeAttribute("src"); }
  clearSpeakState();
}

// ---------- Edge neural TTS (free) ----------
let edgeSession = 0;
let edgeAbort = null;
let ttsAudioEl = null;

function getTtsAudio() {
  if (!ttsAudioEl) {
    ttsAudioEl = document.createElement("audio");
    ttsAudioEl.id = "ttsAudio";
    document.body.appendChild(ttsAudioEl);
  }
  return ttsAudioEl;
}

function edgeVoiceId() {
  return savedVoiceName() || "en-US-ChristopherNeural";
}

function playEdge(btn, text, voiceId) {
  const session = ++edgeSession;
  const audio = getTtsAudio();
  audio.pause();
  const v = voiceId || edgeVoiceId();
  audio.onended = () => { if (session === edgeSession) clearSpeakState(); };
  audio.onerror = () => { if (session === edgeSession) { toast("Speech playback failed", "error"); clearSpeakState(); } };
  audio.src = "/api/tts?text=" + encodeURIComponent(text) + "&voice=" + encodeURIComponent(v);
  audio.play().catch(() => {});
}

function playEdgeNoBtn(text) {
  const session = ++edgeSession;
  const audio = getTtsAudio();
  audio.pause();
  const v = edgeVoiceId();
  audio.onended = () => {};
  audio.src = "/api/tts?text=" + encodeURIComponent(text) + "&voice=" + encodeURIComponent(v);
  audio.play().catch(() => {});
}

// ---------- voice chat (speak & hear replies) ----------
let mediaRec = null;
let mediaStream = null;
let voiceChunks = [];
let voiceDiscard = false;
let silenceWatcher = null;
let silenceCtx = null;
let srRec = null;
let srFinalText = "";
let srLive = false;

function voiceOverlay() { return $("#voiceOverlay"); }
function voiceStatus() { return $("#voiceStatus"); }
function voiceSub() { return $("#voiceSub"); }
function voiceLiveText() { return $("#voiceLiveText"); }

function srAvailable() {
  return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

async function startVoiceChat() {
  if (srLive || mediaRec) { stopVoiceChat(); return; }
  stopSpeaking();
  if (srAvailable()) { startSpeechRecognition(); return; }
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    toast("Microphone access was blocked. Allow it and try again.", "error");
    return;
  }
  voiceChunks = [];
  voiceDiscard = false;
  const mime = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/mp4";
  mediaRec = new MediaRecorder(mediaStream, { mimeType: mime });
  mediaRec.ondataavailable = (e) => { if (e.data && e.data.size) voiceChunks.push(e.data); };
  mediaRec.onstop = handleVoiceRecording;
  mediaRec.start();
  voiceOverlay().classList.remove("hidden");
  voiceStatus().textContent = "Listening… speak now";
  voiceSub().textContent = "Stops by itself when you pause — tap the mic to stop early";
  if (voiceLiveText()) voiceLiveText().textContent = "";
  voiceMicBtn().classList.add("live");
  startSilenceWatcher();
  setTimeout(() => { if (mediaRec && mediaRec.state === "recording") stopVoiceChat(); }, 60000);
}

function startSpeechRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec;
  try {
    rec = new SR();
  } catch (e) {
    toast("Speech recognition could not start", "error");
    return;
  }
  rec.lang = "en-US";
  rec.continuous = false;
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  srRec = rec;
  srFinalText = "";
  srLive = true;
  voiceDiscard = false;
  voiceOverlay().classList.remove("hidden");
  voiceStatus().textContent = "Listening… speak now";
  voiceSub().textContent = "It stops on its own when you finish talking";
  if (voiceLiveText()) voiceLiveText().textContent = "";
  voiceMicBtn().classList.add("live");
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      if (res.isFinal) srFinalText += (srFinalText ? " " : "") + res[0].transcript.trim();
      else interim += res[0].transcript;
    }
    const live = voiceLiveText();
    if (live) live.textContent = srFinalText + (interim ? " " + interim : "");
    voiceSub().textContent = srFinalText ? "Tap the mic again when you are done" : "It stops on its own when you finish talking";
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      toast("Microphone access was blocked. Allow it and try again.", "error");
      srLive = false;
      closeVoiceChat();
    } else if (e.error === "network") {
      toast("Speech service unreachable — check your internet", "error");
      srLive = false;
      closeVoiceChat();
    } else if (e.error === "no-speech") {
      voiceSub().textContent = "Hearing nothing — tap the mic and speak";
    }
  };
  rec.onend = () => {
    if (!srLive) return;
    srLive = false;
    srRec = null;
    closeVoiceChat();
    if (voiceDiscard) { voiceDiscard = false; return; }
    const text = srFinalText.trim();
    if (!text) { toast("Did not catch that — try again", "error"); return; }
    putVoiceTextInComposer(text);
  };
  try { rec.start(); } catch (e) { toast("Speech recognition could not start", "error"); srLive = false; closeVoiceChat(); }
}

function startSilenceWatcher() {
  stopSilenceWatcher();
  if (!mediaStream) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    silenceCtx = new Ctx();
    const src = silenceCtx.createMediaStreamSource(mediaStream);
    const analyser = silenceCtx.createAnalyser();
    analyser.fftSize = 2048;
    src.connect(analyser);
    const data = new Float32Array(analyser.fftSize);
    let quiet = 0;
    const QUIET_MS = 1800;
    silenceWatcher = setInterval(() => {
      if (!mediaRec || mediaRec.state !== "recording") { stopSilenceWatcher(); return; }
      analyser.getFloatTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
      const rms = Math.sqrt(sum / data.length);
      if (rms < 0.025) {
        quiet += 200;
        if (quiet >= QUIET_MS - 600 && quiet < QUIET_MS) {
          voiceSub().textContent = "Heard a pause — wrapping up…";
        }
        if (quiet >= QUIET_MS) {
          stopSilenceWatcher();
          stopVoiceChat();
        }
      } else {
        quiet = 0;
      }
    }, 200);
  } catch (e) {
    console.warn("silence watcher failed", e);
    stopSilenceWatcher();
  }
}

function stopSilenceWatcher() {
  if (silenceWatcher) { clearInterval(silenceWatcher); silenceWatcher = null; }
  if (silenceCtx) { try { silenceCtx.close(); } catch (e) {} silenceCtx = null; }
}

function voiceMicBtn() { return $("#voiceMicBtn"); }

function stopVoiceChat() {
  if (srLive && srRec) {
    try { srRec.stop(); } catch (e) {}
    return;
  }
  if (mediaRec && mediaRec.state !== "inactive") {
    mediaRec.stop();
  } else {
    closeVoiceChat();
  }
}

function handleVoiceRecording() {
  stopSilenceWatcher();
  const rec = mediaRec;
  const discard = voiceDiscard;
  voiceDiscard = false;
  closeVoiceChat();
  if (discard) {
    if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
    mediaRec = null;
    return;
  }
  if (!voiceChunks.length) return;
  const blob = new Blob(voiceChunks, { type: rec ? rec.mimeType : "audio/webm" });
  mediaRec = null;
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  trimAndConvert(blob).then((clean) => transcribeAudio(clean));
}

async function trimAndConvert(blob) {
  try {
    const arr = await blob.arrayBuffer();
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const audio = await ctx.decodeAudioData(arr);
    const ch = audio.getChannelData(0);
    const N = ch.length;
    let start = 0;
    let end = N - 1;
    const thr = 0.02;
    while (start < N && Math.abs(ch[start]) < thr) start++;
    while (end > start && Math.abs(ch[end]) < thr) end--;
    if (end <= start) { ctx.close(); return blob; }
    const pad = Math.round(audio.sampleRate * 0.3);
    start = Math.max(0, start - pad);
    end = Math.min(N - 1, end + pad);
    const targetRate = 16000;
    const len = Math.max(1, Math.round((end - start + 1) * targetRate / audio.sampleRate));
    const oc = new OfflineAudioContext(1, len, targetRate);
    const src = oc.createBufferSource();
    src.buffer = audio;
    src.connect(oc.destination);
    src.start(0, start / audio.sampleRate, (end - start + 1) / audio.sampleRate);
    const rendered = await oc.startRendering();
    const data = rendered.getChannelData(0);
    const buf = new ArrayBuffer(44 + data.length * 2);
    const dv = new DataView(buf);
    const wstr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
    wstr(0, "RIFF"); dv.setUint32(4, 36 + data.length * 2, true); wstr(8, "WAVE");
    wstr(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
    dv.setUint16(22, 1, true); dv.setUint32(24, targetRate, true);
    dv.setUint32(28, targetRate * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    wstr(36, "data"); dv.setUint32(40, data.length * 2, true);
    for (let i = 0; i < data.length; i++) {
      const s = Math.max(-1, Math.min(1, data[i]));
      dv.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    ctx.close();
    return new Blob([buf], { type: "audio/wav" });
  } catch (e) {
    console.warn("trim/convert failed, sending raw", e);
    return blob;
  }
}

function closeVoiceChat() {
  voiceOverlay().classList.add("hidden");
  if (voiceMicBtn()) voiceMicBtn().classList.remove("live");
}

async function transcribeAudio(blob) {
  voiceStatus().textContent = "Understanding you…";
  voiceSub().textContent = "&nbsp;";
  voiceOverlay().classList.remove("hidden");
  const fd = new FormData();
  const ext = blob.type.includes("wav") ? ".wav" : blob.type.includes("webm") ? ".webm" : ".m4a";
  fd.append("audio", blob, "speech" + ext);
  try {
    const j = await api("/api/voice/transcribe", { method: "POST", body: fd });
    if (!j.text) {
      toast("Did not catch that - try again", "error");
      closeVoiceChat();
      return;
    }
    closeVoiceChat();
    putVoiceTextInComposer(j.text);
  } catch (e) {
    toast(e.message || "Transcription failed", "error");
    closeVoiceChat();
  }
}

function putVoiceTextInComposer(text) {
  state.voiceSend = true;
  const inp = $("#input");
  inp.value = text;
  autoGrow();
  inp.focus();
  toast("Text ready — press Enter to send, or edit it first");
}

function speakVoiceReply(replyText) {
  const plain = mdToPlain(replyText);
  if (!plain) return;
  if (state.edgeTts) playEdgeNoBtn(plain);
  else if (speechAvailable()) speakBrowser(plain);
}

function speakBrowser(text) {
  if (!speechAvailable() || !text) return;
  speechSynthesis.cancel();
  const voice = pickVoice();
  const parts = chunkSpeech(text);
  let i = 0;
  const next = () => {
    if (i >= parts.length) return;
    const u = new SpeechSynthesisUtterance(parts[i++]);
    if (voice) { u.voice = voice; u.lang = voice.lang || "en-US"; }
    u.rate = 0.85;
    u.pitch = 0.6;
    u.onend = next;
    u.onerror = next;
    speechSynthesis.speak(u);
    if (speechSynthesis.paused) speechSynthesis.resume();
  };
  next();
}

function speakMessage(btn, text) {
  if (speakState && speakState.btn === btn) {
    if (speakState.paused) { resumeSpeech(); return; }
    pauseSpeech();
    return;
  }
  const plain = mdToPlain(text);
  if (!plain) {
    toast("Nothing to read", "error");
    return;
  }
  if (state.edgeTts) {
    stopSpeaking();
    speakState = { btn, chunks: [], idx: 0, mode: "edge" };
    btn.classList.add("on");
    setSpeakLabel(btn, "Pause");
    playEdge(btn, plain);
    return;
  }
  if (!speechAvailable()) {
    toast("Speech is not supported in this browser", "error");
    return;
  }
  speechSynthesis.cancel();
  clearSpeakState();
  const chunks = chunkSpeech(plain);
  const voice = pickVoice();
  if (!voice) {
    toast("No speech voice found on this computer", "error");
    return;
  }
  speakState = { btn, chunks, idx: 0, mode: "speech" };
  btn.classList.add("on");
  setSpeakLabel(btn, "Pause");
  const speakNext = () => {
    if (!speakState || speakState.paused || speakState.idx >= speakState.chunks.length) {
      if (speakState && speakState.idx >= speakState.chunks.length) clearSpeakState();
      return;
    }
    const u = new SpeechSynthesisUtterance(speakState.chunks[speakState.idx++]);
    u.lang = voice.lang || "en-US";
    u.voice = voice;
    u.rate = 0.85;
    u.pitch = 0.6;
    u.volume = 1;
    u.onend = speakNext;
    u.onerror = speakNext;
    speechSynthesis.speak(u);
    if (speechSynthesis.paused) speechSynthesis.resume();
  };
  speakState.next = speakNext;
  speakNext();
}

// ---------- Voice Mode (full-screen continuous conversation) ----------
let vmActive = false;
let vmMuted = false;
let vmListening = false;
let vmSr = null;
let vmSrFinal = "";
let vmSrLive = false;
let vmTurnActive = false;
let vmSilenceTimer = null;
let vmSilenceMs = 0;

// --- Circular audio wave bars ---
const vmBarCount = 48;
const vmBarInner = 52;
const vmBarMaxLen = 35;
let vmBars = [];
let vmBarAnimId = null;
let vmBarAudioCtx = null;
let vmBarAnalyser = null;
let vmBarData = null;
let vmBarStream = null;
let vmIdleAnimId = null;

function vmInitBars() {
  const g = document.getElementById("vmWaveBars");
  if (!g) return;
  g.innerHTML = "";
  vmBars = [];
  for (let i = 0; i < vmBarCount; i++) {
    const angle = (i / vmBarCount) * Math.PI * 2 - Math.PI / 2;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    const x1 = Math.cos(angle) * vmBarInner;
    const y1 = Math.sin(angle) * vmBarInner;
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x1);
    line.setAttribute("y2", y1);
    line.setAttribute("stroke", "#ffffff");
    line.setAttribute("stroke-width", "1.5");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("opacity", "0.3");
    g.appendChild(line);
    vmBars.push({ el: line, angle });
  }
}

function vmStartIdleBars() {
  vmStopBarAnim();
  let t = 0;
  function tick() {
    t += 0.02;
    for (let i = 0; i < vmBars.length; i++) {
      const b = vmBars[i];
      const wave = Math.sin(t + i * 0.15) * 0.5 + 0.5;
      const len = 3 + wave * 6;
      const x1 = Math.cos(b.angle) * vmBarInner;
      const y1 = Math.sin(b.angle) * vmBarInner;
      const x2 = Math.cos(b.angle) * (vmBarInner + len);
      const y2 = Math.sin(b.angle) * (vmBarInner + len);
      b.el.setAttribute("x1", x1);
      b.el.setAttribute("y1", y1);
      b.el.setAttribute("x2", x2);
      b.el.setAttribute("y2", y2);
      b.el.setAttribute("opacity", 0.2 + wave * 0.15);
    }
    vmIdleAnimId = requestAnimationFrame(tick);
  }
  tick();
}

function vmStopIdleBars() {
  if (vmIdleAnimId) { cancelAnimationFrame(vmIdleAnimId); vmIdleAnimId = null; }
}

function vmStartAudioBars() {
  vmStopIdleBars();
  vmStopBarAnim();
  if (!vmBarAudioCtx) {
    try { vmBarAudioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { return; }
  }
  if (vmBarAudioCtx.state === "suspended") vmBarAudioCtx.resume();
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      if (!vmActive) { stream.getTracks().forEach(t => t.stop()); return; }
      const src = vmBarAudioCtx.createMediaStreamSource(stream);
      vmBarAnalyser = vmBarAudioCtx.createAnalyser();
      vmBarAnalyser.fftSize = 128;
      vmBarAnalyser.smoothingTimeConstant = 0.7;
      src.connect(vmBarAnalyser);
      vmBarData = new Uint8Array(vmBarAnalyser.frequencyBinCount);
      vmBarStream = stream;
      vmAnimateBars();
    })
    .catch(() => {});
}

function vmAnimateBars() {
  if (!vmBarAnalyser || !vmBarData) return;
  vmBarAnalyser.getByteFrequencyData(vmBarData);
  const step = Math.floor(vmBarData.length / vmBarCount);
  for (let i = 0; i < vmBarCount; i++) {
    const b = vmBars[i];
    const val = (vmBarData[i * step] || 0) / 255;
    const len = 4 + val * vmBarMaxLen;
    const x1 = Math.cos(b.angle) * vmBarInner;
    const y1 = Math.sin(b.angle) * vmBarInner;
    const x2 = Math.cos(b.angle) * (vmBarInner + len);
    const y2 = Math.sin(b.angle) * (vmBarInner + len);
    b.el.setAttribute("x1", x1);
    b.el.setAttribute("y1", y1);
    b.el.setAttribute("x2", x2);
    b.el.setAttribute("y2", y2);
    b.el.setAttribute("opacity", 0.3 + val * 0.5);
  }
  vmBarAnimId = requestAnimationFrame(vmAnimateBars);
}

function vmStopBarAnim() {
  if (vmBarAnimId) { cancelAnimationFrame(vmBarAnimId); vmBarAnimId = null; }
  if (vmBarStream) { vmBarStream.getTracks().forEach(t => t.stop()); vmBarStream = null; }
  if (vmBarAnalyser) { vmBarAnalyser.disconnect(); vmBarAnalyser = null; }
  if (vmBarAudioCtx && vmBarAudioCtx.state !== "closed") { vmBarAudioCtx.close().catch(() => {}); vmBarAudioCtx = null; }
  for (const b of vmBars) {
    b.el.setAttribute("x2", b.el.getAttribute("x1"));
    b.el.setAttribute("y2", b.el.getAttribute("y1"));
    b.el.setAttribute("opacity", "0.3");
  }
}

function vmOrb() { return $("#vmOrb"); }
function vmStatus() { return $("#vmStatus"); }
function vmSub() { return $("#vmSub"); }
function vmTranscript() { return $("#vmTranscript"); }
function voiceMode() { return $("#voiceMode"); }

function setVmOrbState(state) {
  const orb = vmOrb();
  if (!orb) return;
  orb.classList.remove("listening", "thinking", "speaking", "idle");
  const s = state || "idle";
  orb.classList.add(s);
  // Start/stop audio wave bars based on state
  if (s === "speaking") {
    vmStopIdleBars();
    vmStartAudioBars();
  } else if (s === "idle") {
    vmStopBarAnim();
    vmStartIdleBars();
  } else {
    vmStopBarAnim();
    vmStopIdleBars();
  }
}

function addVmTranscript(role, text) {
  const el = vmTranscript();
  if (!el) return;
  const div = document.createElement("div");
  div.className = "vm-msg " + (role === "user" ? "user" : "assistant");
  div.innerHTML = `<div class="vm-text md">${renderMarkdown(text)}</div>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function openVoiceMode() {
  if (vmActive) return;
  vmActive = true;
  vmMuted = false;
  voiceMode().classList.remove("hidden");
  document.body.style.overflow = "hidden";
  $("#vmCloseBtn").onclick = closeVoiceMode;
  $("#vmInterruptBtn").onclick = interruptVoiceMode;
  $("#vmMuteBtn").onclick = toggleVmMute;
  vmOrb().onclick = () => { if (vmTurnActive) interruptVoiceMode(); else if (!vmListening) startVmListening(); };
  setVmStatus("Ready");
  setVmSub("Tap the orb to start talking");
  updateVmMuteBtn();
  vmInitBars();
  setVmOrbState("idle");
}

function closeVoiceMode() {
  if (!vmActive) return;
  stopVmListening();
  vmStopBarAnim();
  vmStopIdleBars();
  interruptVoiceMode();
  vmActive = false;
  voiceMode().classList.add("hidden");
  document.body.style.overflow = "";
  vmOrb().onclick = null;
  vmTranscript().innerHTML = "";
  setVmOrbState("idle");
}

function interruptVoiceMode() {
  stopVmListening();
  stopSpeaking();
  vmTurnActive = false;
  setVmOrbState("idle");
  setVmStatus("Ready");
  setVmSub("Tap the orb to start talking");
}

function toggleVmMute() {
  vmMuted = !vmMuted;
  if (vmMuted) stopSpeaking();
  updateVmMuteBtn();
}

function updateVmMuteBtn() {
  const btn = $("#vmMuteBtn");
  if (!btn) return;
  if (vmMuted) {
    btn.innerHTML = '<svg><use href="#i-mic"/></svg> Unmute';
    btn.classList.add("muted");
  } else {
    btn.innerHTML = '<svg><use href="#i-mic-off"/></svg> Mute';
    btn.classList.remove("muted");
  }
}

function setVmStatus(text) { const el = vmStatus(); if (el) el.textContent = text; }
function setVmSub(text) { const el = vmSub(); if (el) el.textContent = text; }

async function startVmListening() {
  if (vmListening || vmTurnActive || !srAvailable()) return;
  vmListening = true;
  vmTurnActive = true;
  vmSrFinal = "";
  vmSilenceMs = 0;

  setVmOrbState("listening");
  setVmStatus("Listening…");
  setVmSub("Speak naturally — I'll wait until you're done");

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let rec;
  try { rec = new SR(); } catch (e) { setVmStatus("Speech recognition unavailable"); vmListening = false; vmTurnActive = false; setVmOrbState("idle"); return; }
  rec.lang = "en-US";
  rec.continuous = true;
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  vmSr = rec;
  vmSrFinal = "";
  vmSrLive = true;

  let hasFinalSpeech = false;
  rec.onresult = (e) => {
    let interim = "";
    let newFinal = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      if (res.isFinal) newFinal += (newFinal ? " " : "") + res[0].transcript.trim();
      else interim += res[0].transcript;
    }
    if (newFinal) {
      hasFinalSpeech = true;
      vmSrFinal += (vmSrFinal ? " " : "") + newFinal;
      vmSilenceMs = 0; // reset silence timer on new speech
    }
    setVmSub(vmSrFinal + (interim ? " " + interim : ""));
  };
  rec.onerror = (e) => {
    if (e.error === "not-allowed" || e.error === "service-not-allowed") {
      setVmStatus("Microphone blocked — allow permission");
      stopVmListening();
    } else if (e.error === "network") {
      setVmStatus("Network error — check connection");
      stopVmListening();
    } else if (e.error === "no-speech") {
      // silence detected by browser, our timer handles it
    }
  };
  rec.onend = () => {
    if (!vmSrLive) return;
    vmSrLive = false;
    vmSr = null;
    vmListening = false;
    const text = vmSrFinal.trim();
    if (!text) {
      setVmStatus("Didn't catch that");
      setVmSub("Tap the orb to try again");
      vmTurnActive = false;
      setVmOrbState("idle");
      return;
    }
    processVmTurn(text);
  };
  try { rec.start(); } catch (e) { setVmStatus("Could not start listening"); vmListening = false; vmTurnActive = false; setVmOrbState("idle"); }

  // Silence detection: stop after ~1.8s of silence AFTER user has spoken
  if (vmSilenceTimer) clearInterval(vmSilenceTimer);
  vmSilenceTimer = setInterval(() => {
    if (!vmSrLive) { clearInterval(vmSilenceTimer); return; }
    vmSilenceMs += 200;
    // Stop if: we have captured some final speech AND silence threshold reached
    if (hasFinalSpeech && vmSilenceMs >= 1800) {
      clearInterval(vmSilenceTimer);
      if (vmSr && vmSrLive) { try { vmSr.stop(); } catch (e) {} }
    }
  }, 200);
}

function stopVmListening() {
  if (vmSilenceTimer) { clearInterval(vmSilenceTimer); vmSilenceTimer = null; }
  if (vmSr && vmSrLive) { try { vmSr.stop(); } catch (e) {} vmSrLive = false; vmSr = null; }
  vmListening = false;
}

async function processVmTurn(userText) {
  addVmTranscript("user", userText);
  setVmOrbState("thinking");
  setVmStatus("Thinking…");
  setVmSub("Ultron is formulating a reply");

  let fullReply = "";
  await stream(
    { message: userText, files: [], chat_id: state.chatId, regenerate: false },
    false,
    (reply) => { fullReply = reply; }
  );

  if (!vmActive) return;
  if (!fullReply) {
    setVmOrbState("idle");
    setVmStatus("No reply — tap to try again");
    vmTurnActive = false;
    return;
  }

  addVmTranscript("assistant", fullReply);
  if (!vmMuted && vmActive) {
    setVmOrbState("speaking");
    setVmStatus("Speaking…");
    setVmSub("Ultron is replying");
    speakVoiceReply(fullReply);
    await waitForSpeechEnd();
  }
  if (!vmActive) return;
  vmTurnActive = false;
  // Auto-restart listening for continuous conversation
  if (vmActive && !vmMuted) {
    setTimeout(() => startVmListening(), 300);
  } else {
    setVmOrbState("idle");
    setVmStatus("Ready");
    setVmSub("Tap the orb to continue");
  }
}

function waitForSpeechEnd() {
  return new Promise((resolve) => {
    const check = setInterval(() => {
      const audio = getTtsAudio();
      const isEdgeSpeaking = speakState && speakState.mode === "edge" && audio.src && !audio.ended && !audio.paused;
      const isSpeechSpeaking = speakState && speakState.mode === "speech" && speechSynthesis.speaking;
      if (!speakState || (!isEdgeSpeaking && !isSpeechSpeaking)) {
        clearInterval(check);
        resolve();
      }
    }, 150);
  });
}

// ---------- attachments ----------
$("#attachBtn").addEventListener("click", () => $("#fileInput").click());
$("#fileInput").addEventListener("change", async (e) => {
  const files = Array.from(e.target.files || []);
  e.target.value = "";
  if (!files.length) return;
  
  const overlay = $("#uploadProgressOverlay");
  const progressList = $("#uploadProgressList");
  let uploadCount = 0;
  const totalFiles = files.length;

  // Process files sequentially to keep indices stable
  for (const f of files) {
    uploadCount++;
    // Show overlay with progress
    overlay.classList.remove("hidden");
    progressList.innerHTML = "";
    
    // Add pending attachment with progress
    const pendingIdx = state.attachments.length;
    state.attachments.push({ id: null, name: f.name, kind: f.type.startsWith("image/") ? "image" : "doc", url: null, progress: 0, pending: true });
    renderAttachmentChips();

    const fd = new FormData();
    fd.append("files", f);
    try {
      const xhr = new XMLHttpRequest();
      const response = await new Promise((resolve, reject) => {
        xhr.upload.addEventListener("progress", (evt) => {
          if (evt.lengthComputable) {
            const pct = Math.round((evt.loaded / evt.total) * 100);
            // Update overlay progress
            updateProgressOverlay(uploadCount, totalFiles, f.name, pct);
            // Re-find the attachment by index (in case array shifted)
            if (pendingIdx < state.attachments.length) {
              state.attachments[pendingIdx].progress = pct;
              renderAttachmentChips();
            }
          }
        });
        xhr.addEventListener("load", () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            resolve(xhr.responseText);
          } else {
            reject(new Error(`Upload failed: ${xhr.status}`));
          }
        });
        xhr.addEventListener("error", () => reject(new Error("Network error")));
        xhr.open("POST", "/api/upload");
        xhr.send(fd);
      });
      const j = JSON.parse(response);
      // Replace pending with actual result
      if (pendingIdx < state.attachments.length) {
        state.attachments.splice(pendingIdx, 1);
      }
      j.results?.forEach((r) => {
        if (r.error) { toast(`Upload skipped: ${r.error}`, "error"); return; }
        state.attachments.push({ id: r.id, name: r.name, kind: r.kind, url: r.url });
      });
      renderAttachmentChips();
    } catch (err) {
      if (pendingIdx < state.attachments.length) {
        state.attachments.splice(pendingIdx, 1);
      }
      renderAttachmentChips();
      toast(err.message, "error");
    }
  }
  // Hide overlay when all uploads complete
  overlay.classList.add("hidden");
});

function updateProgressOverlay(current, total, filename, percent) {
  const progressList = $("#uploadProgressList");
  progressList.innerHTML = `
    <div class="upload-progress-item">
      <div class="upload-progress-info">
        <span>${current} / ${total}</span>
        <span>${escapeHtml(filename)}</span>
      </div>
      <div class="upload-progress-bar-wrap">
        <div class="upload-progress-bar" style="width: ${percent}%"></div>
      </div>
    </div>`;
}

function renderAttachmentChips() {
  const wrap = $("#attachmentChips");
  if (!state.attachments.length) { wrap.innerHTML = ""; return; }
  wrap.innerHTML = state.attachments.map((a, i) => {
    const isPending = a.pending;
    const progressHtml = isPending ? `
      <div class="attach-progress">
        <div class="attach-progress-bar" style="width: ${a.progress || 0}%"></div>
      </div>
      <span class="attach-progress-text">${a.progress || 0}%</span>
    ` : "";
    return `
    <div class="attach-new${isPending ? ' attach-pending' : ''}">
      <span class="icon">${a.kind === "image" && a.url ? `<img src="${a.url}" alt="">` : (a.kind === "doc" ? "🞩" : a.name.split(".").pop())}</span>
      <span class="name" title="${escapeHtml(a.name)}">${escapeHtml(a.name)}</span>
      ${progressHtml}
      <button class="rm" data-i="${i}" title="Remove"><svg><use href="#i-close"/></svg></button>
    </div>`;
  }).join("");
  wrap.querySelectorAll(".rm").forEach((btn) => {
    btn.onclick = () => { state.attachments.splice(parseInt(btn.dataset.i, 10), 1); renderAttachmentChips(); };
  });
}

// ---------- memory ----------
async function loadMemory() {
  const items = await api("/api/memory");
  const list = $("#memoryList");
  if (!items.length) {
    list.innerHTML = '<div class="memory-empty">No memories yet. Ultron learns facts from you as you chat, or add one above.</div>';
    return;
  }
  list.innerHTML = items.map((m) => `
    <div class="memory-item" data-id="${m.id}">
      <span class="cat">${escapeHtml(m.category)}</span>
      <span class="content">${escapeHtml(m.content)}</span>
      <button class="edit" title="Edit"><svg><use href="#i-edit"/></svg></button>
      <button class="del" title="Delete"><svg><use href="#i-trash"/></svg></button>
    </div>`).join("");
  list.querySelectorAll(".memory-item").forEach((item) => {
    const id = item.dataset.id;
    item.querySelector(".edit").onclick = () => editMemory(item);
    item.querySelector(".del").onclick = async () => {
      await api(`/api/memory/${id}`, { method: "DELETE" });
      toast("Memory deleted");
      loadMemory();
    };
  });
}

function editMemory(item) {
  const input = $("#memoryInput");
  input.value = item.querySelector(".content").textContent;
  input.focus();
  state.editingMemoryId = parseInt(item.dataset.id, 10);
  $("#memoryAddBtn").style.background = "var(--accent-3)";
}

async function addOrUpdateMemory() {
  const content = $("#memoryInput").value.trim();
  if (!content) return;
  if (state.editingMemoryId) {
    await api(`/api/memory/${state.editingMemoryId}`, { method: "PATCH", body: { content } });
    state.editingMemoryId = null;
    $("#memoryAddBtn").style.background = "";
    toast("Memory updated");
  } else {
    await api("/api/memory", { method: "POST", body: { content } });
    toast("Memory saved");
  }
  $("#memoryInput").value = "";
  loadMemory();
}

// ---------- export / import ----------
function exportData() {
  api("/api/export").then((data) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ultron-backup.json";
    a.click();
    URL.revokeObjectURL(url);
    toast("Backup downloaded");
  });
}

$("#importFile").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    if (data.app !== "ultron") throw new Error("Not an Ultron backup file");
    await api("/api/import", { method: "POST", body: data });
    toast("Chats imported");
    loadSettings();
    listChats();
    if (!state.chatId) newChat();
  } catch (err) {
    toast("Import failed: " + err.message, "error");
  }
});

// ---------- modals ----------
function openModal(sel) {
  $(sel).classList.remove("hidden");
  requestAnimationFrame(() => {
    const first = $(sel).querySelector("input, textarea, select");
    if (first) setTimeout(() => first.focus(), 120);
  });
}
function closeModals() {
  $$(".modal-backdrop").forEach((m) => m.classList.add("hidden"));
}
$$(".modal-backdrop").forEach((backdrop) => {
  backdrop.addEventListener("click", (e) => {
    if (e.target === backdrop) closeModals();
  });
  backdrop.querySelectorAll("[data-close]").forEach((btn) => btn.addEventListener("click", closeModals));
});

// settings modal tabs
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    const panel = tab.dataset.tab;
    $$(".tab-panel, .data-panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === panel));
  });
});

// ---------- search ----------
let searchTimer = null;
$("#searchInput").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(async () => {
    const q = e.target.value.trim();
    if (!q) return listChats();
    const results = await api("/api/search?q=" + encodeURIComponent(q));
    renderChatList(results, state.chatId);
  }, 250);
});

// ---------- provider select ----------
$("#providerSelect").addEventListener("change", async (e) => {
  const value = e.target.value;
  state.currentProvider = value;
  $("#modelDot").classList.remove("pulse");
  void $("#modelDot").offsetWidth;
  $("#modelDot").classList.add("pulse");
  try {
    await api("/api/settings", { method: "POST", body: { provider: value } });
    state.settings.provider = value;
    toast("Provider set to " + value);
  } catch (err) { toast(err.message, "error"); }
});

// ---------- knowledge base ----------
let kbFiles = [];

function kbNamespaceValue() {
  return ($("#kbNamespace").value.trim() || "healthcare");
}

async function loadKbStatus() {
  try {
    const data = await api("/api/kb/status");
    const ns = data.namespaces || {};
    const entries = Object.entries(ns).filter(([name]) => name);
    if (entries.length === 0) {
      $("#kbStatus").textContent = "No knowledge base ingested yet.";
    } else {
      $("#kbStatus").textContent = "Ingested: " + entries.map(([name, count]) => `${name} (${count} chunks)`).join(", ");
    }
  } catch (err) {
    $("#kbStatus").textContent = "Could not load status: " + err.message;
  }
}

function renderKbResults(results) {
  const ul = $("#kbResults");
  ul.innerHTML = "";
  for (const r of results || []) {
    const li = document.createElement("li");
    li.className = r.error ? "err" : "ok";
    li.innerHTML = `<b>${escapeHtml(r.name)}</b><span>${r.error ? r.error : `${r.chunks} chunks`}</span>`;
    ul.appendChild(li);
  }
}

async function ingestKb() {
  const namespace = kbNamespaceValue();
  if (kbFiles.length === 0) {
    toast("Choose or drop files first", "error");
    return;
  }
  $("#kbIngestBtn").disabled = true;
  $("#kbIngestBtn").textContent = "Ingesting...";
  try {
    const fd = new FormData();
    fd.append("namespace", namespace);
    for (const f of kbFiles) fd.append("files", f);
    const data = await api("/api/kb/ingest", { method: "POST", body: fd });
    renderKbResults(data.results);
    kbFiles = [];
    $("#kbFileInput").value = "";
    toast(`Ingested ${data.total_chunks} chunks into '${namespace}'`);
    loadKbStatus();
    loadKbFiles();
  } catch (err) {
    toast("Ingest failed: " + err.message, "error");
  } finally {
    $("#kbIngestBtn").disabled = false;
    $("#kbIngestBtn").textContent = "Ingest files";
  }
}

async function clearKb() {
  const namespace = kbNamespaceValue();
  if (!confirm(`Remove all chunks in namespace '${namespace}'?`)) return;
  try {
    await api("/api/kb/namespace/" + encodeURIComponent(namespace), { method: "DELETE" });
    $("#kbResults").innerHTML = "";
    toast(`Cleared '${namespace}'`);
    loadKbStatus();
    loadKbFiles();
  } catch (err) {
    toast("Clear failed: " + err.message, "error");
  }
}

async function loadKbFiles() {
  const namespace = kbNamespaceValue();
  try {
    const data = await api("/api/kb/files?namespace=" + encodeURIComponent(namespace));
    renderKbFileList(data.files || []);
  } catch (err) {
    $("#kbFileList").innerHTML = `<li class="err-row">Could not load files: ${escapeHtml(err.message)}</li>`;
  }
}

function renderKbFileList(files) {
  const ul = $("#kbFileList");
  ul.innerHTML = "";
  if (!files.length) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="fname muted">No files in this namespace.</span>`;
    ul.appendChild(li);
    return;
  }
  for (const f of files) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="fname" title="${escapeHtml(f.filename)}">${escapeHtml(f.filename)}</span>
      <span class="fcount">${f.chunks} chunks</span>
      <button class="fdel" title="Delete file" aria-label="Delete"><svg><use href="#i-trash"/></svg></button>`;
    li.querySelector(".fdel").onclick = () => deleteKbFile(f.id, f.filename);
    ul.appendChild(li);
  }
}

async function deleteKbFile(id, name) {
  if (!confirm(`Delete '${name}' from the knowledge base?`)) return;
  try {
    await api("/api/files/" + id, { method: "DELETE" });
    toast(`Deleted ${name}`);
    loadKbStatus();
    loadKbFiles();
  } catch (err) {
    toast("Delete failed: " + err.message, "error");
  }
}

function openKb() {
  kbFiles = [];
  $("#kbResults").innerHTML = "";
  $("#kbNamespace").value = state.settings.kb_namespace || "healthcare";
  loadKbStatus();
  loadKbFiles();
  openModal("#kbModal");
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- event wiring ----------
function wire() {
  $("#newChatBtn").onclick = newChat;
  $("#sendBtn").onclick = sendMessage;
  $("#stopBtn").onclick = () => { stopGeneration(); toast("Stopped"); };
  $("#voiceBtn").onclick = startVoiceChat;
  $("#voiceMicBtn").onclick = () => { if (srLive || mediaRec) stopVoiceChat(); };
  $("#voiceCloseBtn").onclick = () => {
    voiceDiscard = true;
    if (srLive && srRec) { try { srRec.abort(); } catch (e) {} }
    else if (mediaRec && mediaRec.state !== "inactive") { try { mediaRec.stop(); } catch (e) {} }
    else closeVoiceChat();
  };
  $("#voiceModeBtn").onclick = () => {
    const btn = $("#voiceModeBtn");
    if (vmActive) { closeVoiceMode(); btn.classList.remove("active"); }
    else { openVoiceMode(); btn.classList.add("active"); }
  };
  $("#vmCloseBtn").onclick = () => {
    closeVoiceMode();
    $("#voiceModeBtn").classList.remove("active");
  };
  $("#settingsBtn").onclick = () => { loadSettings(); populateVoiceSelect(); openModal("#settingsModal"); };
  $("#memoryBtn").onclick = () => { state.editingMemoryId = null; $("#memoryInput").value = ""; $("#memoryAddBtn").style.background = ""; loadMemory(); openModal("#memoryModal"); };
  $("#kbBtn").onclick = openKb;
  $("#kbIngestBtn").onclick = ingestKb;
  $("#kbClearBtn").onclick = clearKb;
  $("#kbNamespace").addEventListener("input", () => { loadKbStatus(); loadKbFiles(); });
  const kbDrop = $("#kbDrop");
  const kbFileInput = $("#kbFileInput");
  kbDrop.onclick = () => kbFileInput.click();
  kbFileInput.addEventListener("change", () => {
    kbFiles = Array.from(kbFileInput.files);
    if (kbFiles.length) toast(`${kbFiles.length} file(s) ready to ingest`);
  });
  kbDrop.addEventListener("dragover", (e) => { e.preventDefault(); kbDrop.classList.add("dragover"); });
  kbDrop.addEventListener("dragleave", () => kbDrop.classList.remove("dragover"));
  kbDrop.addEventListener("drop", (e) => {
    e.preventDefault();
    kbDrop.classList.remove("dragover");
    kbFiles = Array.from(e.dataTransfer.files || []);
    if (kbFiles.length) toast(`${kbFiles.length} file(s) ready to ingest`);
  });
  $("#menuBtn").onclick = () => document.querySelector(".app").classList.toggle("sidebar-open");
  $("#themeToggle").onclick = toggleTheme;
  $("#memoryAddBtn").onclick = addOrUpdateMemory;
  $("#memoryInput").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); addOrUpdateMemory(); } });
  $("#sidebarScrim").onclick = () => document.querySelector(".app").classList.remove("sidebar-open");
  $("#setThemeDark").addEventListener("change", () => {
    setTheme($("#setThemeDark").checked ? "dark" : "light");
  });
  $("#collapseBtn").onclick = () => {
    const app = document.querySelector(".app");
    const hidden = app.classList.toggle("sidebar-hidden");
    localStorage.setItem("ultron-sidebar", hidden ? "1" : "0");
  };
  $("#expandBtn").onclick = () => {
    const app = document.querySelector(".app");
    app.classList.remove("sidebar-hidden");
    localStorage.setItem("ultron-sidebar", "0");
  };
  $("#deleteAllChatsBtn").onclick = async () => {
    if (!confirm("Delete ALL chats? This cannot be undone.")) return;
    try {
      await api("/api/chats/all", { method: "DELETE" });
      toast("All chats deleted");
      state.chatId = null;
      state.messages = [];
      renderConversation([]);
      setEmpty(true);
      await listChats();
    } catch (e) {
      toast(e.message, "error");
    }
  };
  $("#saveSettingsBtn").onclick = async () => {
    await api("/api/settings", { method: "POST", body: readSettingsForm() });
    state.settings = readSettingsForm();
    state.settings.provider = $("#setProvider").value;
    if ($("#setVoice")) localStorage.setItem("ultron-voice", $("#setVoice").value || "");
    loadSettings();
    closeModals();
    toast("Settings saved");
  };
  $("#setVoice").addEventListener("change", () => {
    localStorage.setItem("ultron-voice", $("#setVoice").value || "");
    if ($("#setVoice").value) toast(`Voice set to ${$("#setVoice").value}`);
  });
  $("#exportBtn").onclick = exportData;
  $("#importBtn").onclick = () => $("#importFile").click();

  input.addEventListener("input", autoGrow);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  $$(".chip").forEach((chip) => chip.addEventListener("click", () => {
    input.value = chip.dataset.prompt;
    autoGrow();
    sendMessage();
  }));

  // Esc closes modals
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { if (vmActive) { closeVoiceMode(); $("#voiceModeBtn").classList.remove("active"); } else { closeModals(); } } });
}

// ---------- init ----------
async function init() {
  // theme
  const savedTheme = localStorage.getItem("ultron-theme");
  const s = await loadSettings();
  state.theme = savedTheme || state.settings.theme || "dark";
  setTheme(state.theme);

  wire();
  await listChats();
  if (localStorage.getItem("ultron-sidebar") === "1") document.querySelector(".app").classList.add("sidebar-hidden");
  stopSpeaking();
  populateVoiceSelect();
  if (speechAvailable()) {
    loadVoices();
    if (!_ttsVoices || !_ttsVoices.length) {
      speechSynthesis.onvoiceschanged = () => { loadVoices(); };
      setTimeout(loadVoices, 250);
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch((e) => toast(e.message, "error"));
});