const app = document.getElementById("app");

const state = {
  view: "start",
  intro: 0,
  token: localStorage.getItem("hmo_token") || "",
  user: null,
  settings: { detector: "hog", scan_device: "auto" },
  message: "",
  busy: false,
  jobId: null,
  result: null,
  history: [],
  selectedFile: null,
  stream: null,
  recorder: null,
  chunks: [],
  recordingStartedAt: 0,
  timerId: null,
  pollId: null,
  liveFallbackFile: null,
  translationMode: "en",
};

const introSlides = [
  {
    kind: "logo",
    title: "HearMeOut",
    copy: "AI-powered lip reading assistant",
  },
  {
    kind: "lips",
    title: "Read Lips with AI",
    copy: "Transform lip movements into accurate text using advanced AI technology.",
  },
  {
    kind: "upload",
    title: "Upload or Record",
    copy: "Upload a video or record live to start real-time lip reading.",
  },
  {
    kind: "clock",
    title: "Get Instant Results",
    copy: "Receive fast and clear text predictions within seconds.",
  },
];

function icon(name, className = "nav-icon") {
  return `<svg class="${className}" aria-hidden="true"><use href="#${name}"></use></svg>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setView(view) {
  stopLiveCamera();
  clearJobPolling();
  state.message = "";
  state.view = view;
  render();
}

function clearJobPolling() {
  if (state.pollId) {
    clearTimeout(state.pollId);
    state.pollId = null;
  }
}

function render() {
  if (state.view !== "live") stopLiveCamera();

  const pages = {
    start: renderStart,
    signup: renderSignup,
    login: renderLogin,
    success: renderSuccess,
    home: renderHome,
    live: renderLive,
    upload: renderUpload,
    processing: renderProcessing,
    result: renderResult,
    history: renderHistory,
    settings: renderSettings,
  };
  app.innerHTML = pages[state.view]();

  if (state.view === "live") mountLiveCamera();
  if (state.view === "history") loadHistory();
  if (state.view === "settings") loadSettings();
}

async function api(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.body);
  }
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

function renderStart() {
  const slide = introSlides[state.intro];
  const isLast = state.intro === introSlides.length - 1;
  return `
    <main class="screen center-screen">
      ${slide.kind === "logo" ? `
        <img class="logo-large" src="/assets/logo.svg" alt="HearMeOut logo">
      ` : `
        <div class="intro-art">${introArt(slide.kind)}</div>
      `}
      <section class="intro-copy">
        <h1>${escapeHtml(slide.title)}</h1>
        <p>${escapeHtml(slide.copy)}</p>
      </section>
      <div class="dots" aria-hidden="true">
        ${introSlides.map((_, i) => `<span class="dot ${i === state.intro ? "active" : ""}"></span>`).join("")}
      </div>
      ${isLast ? `
        <div class="button-row">
          <button class="primary-btn" onclick="setView('signup')">Sign Up</button>
          <button class="secondary-btn" onclick="setView('login')">Login</button>
        </div>
      ` : `
        <button class="primary-btn" onclick="nextIntro()">${state.intro === 0 ? "Start" : "Next"}</button>
        <button class="ghost-btn" onclick="skipIntro()">Skip</button>
      `}
    </main>
  `;
}

function introArt(kind) {
  if (kind === "lips") {
    return `
      <svg viewBox="0 0 100 100">
        <path d="M19 50c13-22 24-27 35-16 7 6 13 6 20 0 12-11 22-6 36 16-20 0-26-8-36-6-7 2-13 2-20 0-10-2-16 6-35 6z" fill="rgba(65,108,255,.75)" stroke="none"/>
        <path d="M20 58c20 2 28 12 46 12s27-10 45-12c-15 20-29 27-45 27S35 78 20 58z" fill="rgba(19,184,244,.75)" stroke="none"/>
        <path d="M10 55c10 0 10-18 20-18s10 36 20 36 10-38 20-38 10 32 20 32" stroke="#fff"/>
      </svg>`;
  }
  if (kind === "upload") {
    return `
      <svg viewBox="0 0 100 100">
        <path d="M30 16h28l18 18v50H30z"/>
        <path d="M58 16v18h18"/>
        <path d="m45 40 20 12-20 12z"/>
        <circle cx="75" cy="74" r="14"/>
        <path d="M75 81V67M69 73l6-6 6 6"/>
      </svg>`;
  }
  return `
    <svg viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="30"/>
      <path d="M50 29v23l17 9"/>
      <path d="M21 68H9M20 50H8M25 33H14M70 70l17 17M83 62l-9 9"/>
    </svg>`;
}

function nextIntro() {
  state.intro = Math.min(introSlides.length - 1, state.intro + 1);
  render();
}

function skipIntro() {
  state.intro = introSlides.length - 1;
  render();
}

function renderSignup() {
  return `
    <main class="screen auth-screen">
      <section class="auth-card">
        <h1>Sign Up</h1>
        <p class="subtitle">Enter your details below and create a free account.</p>
        <form onsubmit="submitSignup(event)">
          <div class="field">
            <label for="signup-name">Name</label>
            <input id="signup-name" name="display_name" autocomplete="name" placeholder="Kristin">
          </div>
          <div class="field">
            <label for="signup-email">Your Email</label>
            <input id="signup-email" name="email" type="email" autocomplete="email" required placeholder="you@example.com">
          </div>
          <div class="field">
            <label for="signup-password">Password</label>
            <input id="signup-password" name="password" type="password" autocomplete="new-password" required minlength="6">
          </div>
          <p class="status-text">${escapeHtml(state.message)}</p>
          <button class="primary-btn full" type="submit">Create account</button>
        </form>
        <p class="small-copy">By creating an account you agree to use this local prototype responsibly.</p>
        <p class="small-copy">Already have an account? <button class="link-btn" onclick="setView('login')">Log in</button></p>
      </section>
    </main>
  `;
}

function renderLogin() {
  return `
    <main class="screen auth-screen">
      <section class="auth-card">
        <h1>Log IN</h1>
        <p class="subtitle">Use the account saved on this laptop.</p>
        <form onsubmit="submitLogin(event)">
          <div class="field">
            <label for="login-email">Your Email</label>
            <input id="login-email" name="email" type="email" autocomplete="email" required placeholder="you@example.com">
          </div>
          <div class="field">
            <label for="login-password">Password</label>
            <input id="login-password" name="password" type="password" autocomplete="current-password" required>
          </div>
          <p class="status-text">${escapeHtml(state.message)}</p>
          <button class="primary-btn full" type="submit">Log In</button>
        </form>
        <p class="small-copy">Do not have an account? <button class="link-btn" onclick="setView('signup')">Sign up</button></p>
      </section>
    </main>
  `;
}

async function submitSignup(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    const data = await api("/api/signup", {
      method: "POST",
      body: {
        display_name: form.get("display_name"),
        email: form.get("email"),
        password: form.get("password"),
      },
    });
    acceptAuth(data);
  } catch (error) {
    state.message = error.message;
    render();
  }
}

async function submitLogin(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    const data = await api("/api/login", {
      method: "POST",
      body: {
        email: form.get("email"),
        password: form.get("password"),
      },
    });
    acceptAuth(data);
  } catch (error) {
    state.message = error.message;
    render();
  }
}

function acceptAuth(data) {
  state.token = data.token;
  state.user = data.user;
  localStorage.setItem("hmo_token", state.token);
  setView("success");
}

function renderSuccess() {
  return `
    <main class="screen center-screen">
      <section class="success-card">
        <div class="success-mark">${icon("i-check")}</div>
        <h1>Success</h1>
        <p class="subtitle">Congratulations, you have completed your registration.</p>
        <button class="primary-btn full" onclick="setView('home')">Done</button>
      </section>
    </main>
  `;
}

function greetingName() {
  const name = state.user?.display_name || "there";
  return name.split(" ")[0] || name;
}

function renderHome() {
  return `
    <main class="screen">
      <section class="home-header">
        <div>
          <h1 class="page-title">Hi, ${escapeHtml(greetingName())}</h1>
          <p class="subtitle">Analyze lip movements quickly and accurately using AI-powered speech recognition.</p>
        </div>
        <div class="avatar">${escapeHtml(greetingName()[0] || "H")}</div>
      </section>
      <section class="action-list">
        <button class="action-card" onclick="setView('live')">
          ${icon("i-record", "action-icon")}
          <span><h2>Record Live</h2><p>Capture lip movements from the phone camera.</p></span>
        </button>
        <button class="action-card" onclick="setView('upload')">
          ${icon("i-upload", "action-icon")}
          <span><h2>Upload Video</h2><p>Analyze speech from uploaded videos.</p></span>
        </button>
        <button class="action-card" onclick="setView('history')">
          ${icon("i-history", "action-icon")}
          <span><h2>History</h2><p>View your saved analyses and results.</p></span>
        </button>
      </section>
      <section class="tips-card">
        <h2>Tips for Better Accuracy</h2>
        <ul>
          <li>Use good lighting</li>
          <li>Keep your face centered</li>
          <li>Avoid excessive head movement</li>
          <li>Speak clearly and slowly</li>
        </ul>
      </section>
    </main>
    ${bottomNav("home")}
  `;
}

function renderLive() {
  return `
    <main class="screen">
      <div class="topbar">
        <button class="ghost-btn" onclick="setView('home')" aria-label="Back">${icon("i-back", "back-icon")}</button>
        <div>
          <h1 class="page-title">Live Recording</h1>
          <p class="subtitle">Analyze lip movements with AI.</p>
        </div>
      </div>
      <section class="camera-frame" id="camera-wrap">
        <video id="camera-preview" autoplay muted playsinline></video>
        <div class="camera-overlay">
          <span class="record-pill"><span class="record-dot" id="record-dot"></span><span id="record-status">Mouth Detection Active</span></span>
        </div>
      </section>
      <p class="status-text" id="live-message"></p>
      <section class="record-tips">
        <h2>Recording Tips</h2>
        <ul>
          <li>Use good lighting</li>
          <li>Keep your face centered</li>
          <li>Hold the camera steady</li>
          <li>Speak clearly and slowly</li>
        </ul>
      </section>
      <button id="capture-button" class="capture-button" onclick="toggleRecording()" aria-label="Start recording"><span></span></button>
      <section id="camera-fallback" class="upload-drop hidden">
        <p>Camera preview needs a secure browser context. Use the camera picker below to record and send a clip.</p>
        <input id="live-file" type="file" accept="video/*" capture="user" onchange="handleLiveFallbackFile(event)">
      </section>
    </main>
  `;
}

async function mountLiveCamera() {
  const video = document.getElementById("camera-preview");
  const fallback = document.getElementById("camera-fallback");
  const message = document.getElementById("live-message");
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    fallback?.classList.remove("hidden");
    if (message) message.textContent = "Use the camera picker on this browser.";
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 720 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = state.stream;
  } catch (error) {
    fallback?.classList.remove("hidden");
    if (message) message.textContent = "Camera preview is blocked. Use the camera picker below.";
  }
}

function stopLiveCamera() {
  if (state.timerId) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
  if (state.recorder && state.recorder.state !== "inactive") {
    try { state.recorder.stop(); } catch (_) { /* ignore */ }
  }
  state.recorder = null;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
}

function toggleRecording() {
  if (state.recorder && state.recorder.state === "recording") {
    state.recorder.stop();
    return;
  }
  if (!state.stream) {
    const message = document.getElementById("live-message");
    if (message) message.textContent = "Camera is not ready yet.";
    return;
  }
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
    ? "video/webm;codecs=vp8"
    : "video/webm";
  state.chunks = [];
  state.recorder = new MediaRecorder(state.stream, { mimeType: mime });
  state.recorder.ondataavailable = (event) => {
    if (event.data.size > 0) state.chunks.push(event.data);
  };
  state.recorder.onstop = () => {
    const blob = new Blob(state.chunks, { type: "video/webm" });
    const file = new File([blob], `live-${Date.now()}.webm`, { type: "video/webm" });
    stopLiveCamera();
    startPrediction(file, "live");
  };
  state.recorder.start();
  state.recordingStartedAt = Date.now();
  document.getElementById("capture-button")?.classList.add("recording");
  document.getElementById("record-dot")?.classList.add("live");
  const status = document.getElementById("record-status");
  state.timerId = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.recordingStartedAt) / 1000);
    const mm = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const ss = String(elapsed % 60).padStart(2, "0");
    if (status) status.textContent = `${mm}:${ss}`;
  }, 250);
}

function handleLiveFallbackFile(event) {
  const file = event.target.files?.[0];
  if (file) startPrediction(file, "live");
}

function renderUpload() {
  return `
    <main class="screen">
      <div class="topbar">
        <button class="ghost-btn" onclick="setView('home')" aria-label="Back">${icon("i-back", "back-icon")}</button>
        <div>
          <h1 class="page-title">Upload Video</h1>
          <p class="subtitle">Send a prerecorded clip to the laptop model.</p>
        </div>
      </div>
      <section class="upload-drop">
        ${icon("i-upload", "action-icon")}
        <p>Select a clear video of the speaker facing the camera.</p>
        <input type="file" id="video-file" accept="video/*" onchange="handleUploadFile(event)">
        <button class="primary-btn full" onclick="submitSelectedFile()">Analyze Video</button>
      </section>
      <section class="tips-card">
        <h2>Tips for Better Accuracy</h2>
        <ul>
          <li>Use short clips when possible</li>
          <li>Keep the mouth visible</li>
          <li>Avoid strong shadows</li>
        </ul>
      </section>
      <p class="status-text">${escapeHtml(state.message)}</p>
    </main>
  `;
}

function handleUploadFile(event) {
  state.selectedFile = event.target.files?.[0] || null;
}

function submitSelectedFile() {
  if (!state.selectedFile) {
    state.message = "Choose a video first.";
    render();
    return;
  }
  startPrediction(state.selectedFile, "upload");
}

async function startPrediction(file, source) {
  clearJobPolling();
  state.message = "";
  state.result = null;
  setView("processing");
  try {
    const body = new FormData();
    body.append("video", file, file.name || "video.webm");
    body.append("source", source);
    body.append("detector", state.settings.detector || "hog");
    const data = await api("/api/predict/upload", { method: "POST", body });
    state.jobId = data.job_id;
    pollJob();
  } catch (error) {
    state.message = error.message;
    setView(source === "live" ? "live" : "upload");
  }
}

function renderProcessing() {
  const message = state.message || "Analyzing lip movements...";
  return `
    <main class="screen processing-screen center-screen">
      <section>
        <h1>Processing Video</h1>
        <p>Analyzing lip movements...</p>
      </section>
      <div class="spinner" aria-hidden="true"></div>
      <div class="soft-btn">AI Analyzing ...</div>
      <section class="progress-card">
        <ul class="progress-list">
          <li><span class="tiny-check">✓</span>Extracting frames...</li>
          <li><span class="tiny-check">✓</span>Detecting mouth movement...</li>
          <li><span class="tiny-check">✓</span>Processing speech patterns...</li>
          <li><span class="tiny-check">✓</span>${escapeHtml(message)}</li>
        </ul>
      </section>
      <p id="progress-label">${escapeHtml(state.progressLabel || "0% completed")}</p>
    </main>
  `;
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const data = await api(`/api/jobs/${state.jobId}`);
    const job = data.job;
    state.progressLabel = `${job.progress || 0}% completed`;
    state.message = job.message || "";
    if (state.view === "processing") render();
    if (job.status === "succeeded") {
      clearJobPolling();
      state.result = job.result;
      state.translationMode = "en";
      setView("result");
      return;
    }
    if (job.status === "failed") {
      clearJobPolling();
      state.message = job.error || "Prediction failed.";
      renderProcessingFailure();
      return;
    }
    state.pollId = setTimeout(pollJob, 1600);
  } catch (error) {
    state.message = error.message;
    state.pollId = setTimeout(pollJob, 2500);
  }
}

function renderProcessingFailure() {
  app.innerHTML = `
    <main class="screen center-screen">
      <section class="success-card">
        <h1>Processing Failed</h1>
        <p class="subtitle">${escapeHtml(state.message)}</p>
        <button class="primary-btn full" onclick="setView('home')">Back Home</button>
      </section>
    </main>
  `;
}

function renderResult() {
  const result = state.result || {};
  const predicted = result.predicted_text || "";
  const shownText = state.translationMode === "ar" && result.arabic_text
    ? result.arabic_text
    : predicted;
  const confidence = typeof result.confidence === "number" ? result.confidence : null;
  return `
    <main class="screen result-screen">
      <div class="topbar">
        <button class="ghost-btn" onclick="setView('home')" aria-label="Back">${icon("i-back", "back-icon")}</button>
        <div>
          <h1 class="page-title">Prediction Result</h1>
          <p class="subtitle">Lip analysis completed successfully.</p>
        </div>
      </div>
      <section class="result-card">
        <h2>Recorded Video</h2>
        <div class="video-box"><div class="center-screen" style="min-height:100%;padding:0">${icon("i-record", "action-icon")}</div></div>
        <h2>Predicted Speech</h2>
        <div class="text-box" dir="${state.translationMode === "ar" ? "rtl" : "ltr"}">${escapeHtml(shownText)}</div>
        <div class="confidence-row">
          <h2>Confidence Score</h2>
          ${confidence === null ? `
            <p class="muted-dark">Confidence is not exposed by this AV-HuBERT decoder.</p>
          ` : `
            <div class="meter"><span style="width:${Math.round(confidence * 100)}%"></span></div>
            <strong>${Math.round(confidence * 100)}%</strong>
          `}
        </div>
      </section>
      <div class="button-row">
        <button class="primary-btn" onclick="toastSaved()">Save Result</button>
        <button class="secondary-btn" onclick="setView('upload')">Try again</button>
      </div>
      <button class="secondary-btn full" style="margin:12px 0" onclick="exportResult()">Export Text</button>
      <section class="result-card">
        <h2>Translation</h2>
        <p class="muted-dark">Translate predicted speech from English to Arabic on the laptop.</p>
        <div class="translate-tabs">
          <button class="${state.translationMode === "en" ? "active" : ""}" onclick="showEnglish()">English</button>
          <button class="${state.translationMode === "ar" ? "active" : ""}" onclick="translateArabic()">Arabic</button>
        </div>
        <p class="status-text">${escapeHtml(state.message)}</p>
      </section>
      <section class="result-card">
        <h2>Text to Speech</h2>
        <p class="muted-dark">Audio playback uses the Android browser's built-in speech engine.</p>
        <button class="primary-btn full" onclick="speakResult()">Play Audio</button>
      </section>
    </main>
  `;
}

function toastSaved() {
  state.message = "Already saved to history.";
  render();
}

function exportResult() {
  const result = state.result || {};
  const lines = [
    "HearMeOut Prediction",
    "",
    result.predicted_text || "",
    "",
    result.arabic_text ? `Arabic: ${result.arabic_text}` : "",
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "hearmeout-prediction.txt";
  link.click();
  URL.revokeObjectURL(url);
}

function showEnglish() {
  state.translationMode = "en";
  state.message = "";
  render();
}

async function translateArabic() {
  if (!state.result?.predicted_text) return;
  if (state.result.arabic_text) {
    state.translationMode = "ar";
    render();
    return;
  }
  state.message = "Loading Arabic translation...";
  render();
  try {
    const data = await api("/api/translate", {
      method: "POST",
      body: {
        text: state.result.predicted_text,
        prediction_id: state.result.id,
      },
    });
    state.result.arabic_text = data.arabic_text;
    state.translationMode = "ar";
    state.message = "";
  } catch (error) {
    state.message = error.message;
  }
  render();
}

function speakResult() {
  const result = state.result || {};
  const text = state.translationMode === "ar" && result.arabic_text
    ? result.arabic_text
    : result.predicted_text;
  if (!text || !window.speechSynthesis) {
    state.message = "Speech playback is not available in this browser.";
    render();
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = state.translationMode === "ar" ? "ar" : "en-US";
  window.speechSynthesis.speak(utterance);
}

function renderHistory() {
  return `
    <main class="screen">
      <h1 class="page-title">Prediction History</h1>
      <p class="subtitle">View and manage your previous analyses.</p>
      <section id="history-container">
        ${historyContent()}
      </section>
    </main>
    ${bottomNav("history")}
  `;
}

function historyContent() {
  if (!state.history.length) {
    return `
      <div class="empty-state">
        <div>
          <div class="empty-icon">${icon("i-history", "action-icon")}</div>
          <h2>No Predictions Yet</h2>
          <p>Your analyzed videos and predictions will appear here.</p>
        </div>
      </div>
    `;
  }
  return `
    <div class="history-list">
      ${state.history.map((item) => `
        <article class="history-card">
          <h3>${escapeHtml(new Date(item.created_at).toLocaleString())}</h3>
          <p>${escapeHtml(item.predicted_text)}</p>
          ${item.arabic_text ? `<p dir="rtl">${escapeHtml(item.arabic_text)}</p>` : ""}
          <button class="danger-btn" onclick="deleteHistory(${item.id})">Delete</button>
        </article>
      `).join("")}
    </div>
  `;
}

async function loadHistory() {
  try {
    const data = await api("/api/history");
    state.history = data.items || [];
    const container = document.getElementById("history-container");
    if (container) container.innerHTML = historyContent();
  } catch (error) {
    state.message = error.message;
  }
}

async function deleteHistory(id) {
  try {
    await api(`/api/history/${id}`, { method: "DELETE" });
    state.history = state.history.filter((item) => item.id !== id);
    render();
  } catch (error) {
    state.message = error.message;
    render();
  }
}

function renderSettings() {
  return `
    <main class="screen">
      <h1 class="page-title">Settings</h1>
      <p class="subtitle">Choose how the laptop should process videos.</p>
      <section class="settings-card">
        <div class="field">
          <label for="detector">Mouth detector</label>
          <select id="detector" onchange="saveSettings()">
            <option value="hog" ${state.settings.detector === "hog" ? "selected" : ""}>HOG CPU detector</option>
            <option value="fa" ${state.settings.detector === "fa" ? "selected" : ""}>Face Alignment GPU detector</option>
          </select>
        </div>
        <div class="field">
          <label for="scan-device">GPU detector device</label>
          <select id="scan-device" onchange="saveSettings()">
            <option value="auto" ${state.settings.scan_device === "auto" ? "selected" : ""}>Auto</option>
            <option value="cpu" ${state.settings.scan_device === "cpu" ? "selected" : ""}>CPU</option>
            <option value="cuda" ${state.settings.scan_device === "cuda" ? "selected" : ""}>CUDA</option>
          </select>
        </div>
        <button class="secondary-btn full" onclick="logout()">Log out</button>
      </section>
      <p class="server-note">The account database and prediction history are stored on this laptop in <strong>mobile_app/data/hearmeout.db</strong>.</p>
    </main>
    ${bottomNav("settings")}
  `;
}

async function loadSettings() {
  try {
    const data = await api("/api/settings");
    state.settings = data.settings || state.settings;
  } catch (_) {
    /* keep defaults */
  }
}

async function saveSettings() {
  const detector = document.getElementById("detector")?.value || "hog";
  const scanDevice = document.getElementById("scan-device")?.value || "auto";
  state.settings = { detector, scan_device: scanDevice };
  try {
    const data = await api("/api/settings", {
      method: "POST",
      body: state.settings,
    });
    state.settings = data.settings || state.settings;
  } catch (error) {
    state.message = error.message;
  }
}

function logout() {
  localStorage.removeItem("hmo_token");
  state.token = "";
  state.user = null;
  state.history = [];
  state.result = null;
  setView("start");
}

function bottomNav(active) {
  return `
    <nav class="bottom-nav" aria-label="Main navigation">
      <button class="${active === "home" ? "active" : ""}" onclick="setView('home')">${icon("i-home")}Home</button>
      <button class="${active === "history" ? "active" : ""}" onclick="setView('history')">${icon("i-history")}History</button>
      <button class="${active === "settings" ? "active" : ""}" onclick="setView('settings')">${icon("i-settings")}Settings</button>
    </nav>
  `;
}

async function boot() {
  if (!state.token) {
    render();
    return;
  }
  try {
    const data = await api("/api/me");
    state.user = data.user;
    state.view = "home";
    await loadSettings();
  } catch (_) {
    localStorage.removeItem("hmo_token");
    state.token = "";
    state.user = null;
    state.view = "start";
  }
  render();
}

window.setView = setView;
window.nextIntro = nextIntro;
window.skipIntro = skipIntro;
window.submitSignup = submitSignup;
window.submitLogin = submitLogin;
window.toggleRecording = toggleRecording;
window.handleLiveFallbackFile = handleLiveFallbackFile;
window.handleUploadFile = handleUploadFile;
window.submitSelectedFile = submitSelectedFile;
window.toastSaved = toastSaved;
window.exportResult = exportResult;
window.showEnglish = showEnglish;
window.translateArabic = translateArabic;
window.speakResult = speakResult;
window.deleteHistory = deleteHistory;
window.saveSettings = saveSettings;
window.logout = logout;

boot();
