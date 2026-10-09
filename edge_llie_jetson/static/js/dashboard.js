/**
 * EdgeLLIE - Dashboard Real-Time Telemetry & Input Controller (Dynamic Light/Dark Theme, VRAM & FPS)
 */

// Global State
let ws = null;
let currentWaveformData = [];
let canvas = null;
let ctx = null;
let currentActiveWeight = "";
let currentSourceType = "sample";

// Client Camera Webcam State
let clientMediaStream = null;
let clientCameraWs = null;
let isClientCameraActive = false;
let clientFrameSending = false;
let clientFpsTracker = { lastTime: performance.now(), frames: 0, fps: 30.0 };

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initCanvas();
    initWebSocket();
    initControls();
    loadAvailableWeights();
    startCanvasRenderLoop();
});

// Theme Management (Light / Dark Theme)
function initTheme() {
    const savedTheme = localStorage.getItem("app-theme") || "dark";
    applyTheme(savedTheme);

    const btnTheme = document.getElementById("btnThemeToggle");
    if (btnTheme) {
        btnTheme.addEventListener("click", () => {
            const currentTheme = document.body.classList.contains("light-theme") ? "light" : "dark";
            const nextTheme = currentTheme === "light" ? "dark" : "light";
            applyTheme(nextTheme);
            localStorage.setItem("app-theme", nextTheme);
        });
    }
}

function applyTheme(theme) {
    const body = document.body;
    const themeIcon = document.getElementById("themeIcon");
    const themeText = document.getElementById("themeText");

    if (theme === "light") {
        body.classList.remove("dark-theme");
        body.classList.add("light-theme");
        if (themeIcon) themeIcon.textContent = "☀️";
        if (themeText) themeText.textContent = "Light";
    } else {
        body.classList.remove("light-theme");
        body.classList.add("dark-theme");
        if (themeIcon) themeIcon.textContent = "🌙";
        if (themeText) themeText.textContent = "Dark";
    }
}

// Canvas Setup
function initCanvas() {
    canvas = document.getElementById("waveformCanvas");
    if (!canvas) return;
    ctx = canvas.getContext("2d");
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}

// Canvas Render Loop (Smooth Oscilloscope at 60 FPS)
function startCanvasRenderLoop() {
    function render() {
        drawWaveform();
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

function drawWaveform() {
    if (!ctx || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);

    const isLight = document.body.classList.contains("light-theme");
    const gridColor = isLight ? "#f1f5f9" : "#162033";
    const zeroLineColor = isLight ? "#e2e8f0" : "#25334d";
    const lineColor = isLight ? "#00a896" : "#22d3ee";
    const gradTop = isLight ? "rgba(6, 182, 212, 0.35)" : "rgba(34, 211, 238, 0.30)";
    const gradMid = isLight ? "rgba(15, 118, 110, 0.12)" : "rgba(45, 212, 191, 0.08)";
    const gradBot = isLight ? "rgba(6, 182, 212, 0.35)" : "rgba(34, 211, 238, 0.30)";

    const leftPad = 42;
    const plotW = w - leftPad - 12;
    const midY = h / 2;

    // Draw Grid Lines (at +1.0, +0.5, 0.0, -0.5, -1.0)
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1;
    
    const gridYValues = [
        midY - (h * 0.38), // +1.0
        midY - (h * 0.19), // +0.5
        midY,              //  0.0
        midY + (h * 0.19), // -0.5
        midY + (h * 0.38)  // -1.0
    ];

    gridYValues.forEach(gy => {
        ctx.beginPath();
        ctx.moveTo(leftPad, gy);
        ctx.lineTo(w - 10, gy);
        ctx.stroke();
    });

    // Center zero reference line
    ctx.strokeStyle = zeroLineColor;
    ctx.beginPath();
    ctx.moveTo(leftPad, midY);
    ctx.lineTo(w - 10, midY);
    ctx.stroke();

    if (!currentWaveformData || currentWaveformData.length === 0) return;

    const data = currentWaveformData;
    const step = plotW / (data.length - 1);

    // Gradient Fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, gradTop);
    grad.addColorStop(0.5, gradMid);
    grad.addColorStop(1, gradBot);

    // Draw filled polygon
    ctx.beginPath();
    ctx.moveTo(leftPad, midY);
    for (let i = 0; i < data.length; i++) {
        const x = leftPad + i * step;
        const val = Math.max(-1.0, Math.min(1.0, data[i]));
        const y = midY - val * (h * 0.38);
        ctx.lineTo(x, y);
    }
    ctx.lineTo(leftPad + (data.length - 1) * step, midY);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Draw Waveform Line
    ctx.beginPath();
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 2.0;
    ctx.lineJoin = "round";

    if (!isLight) {
        ctx.shadowColor = "rgba(34, 211, 238, 0.6)";
        ctx.shadowBlur = 6;
    } else {
        ctx.shadowBlur = 0;
    }

    for (let i = 0; i < data.length; i++) {
        const x = leftPad + i * step;
        const val = Math.max(-1.0, Math.min(1.0, data[i]));
        const y = midY - val * (h * 0.38);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.shadowBlur = 0;
}

// WebSocket Connection
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("[WS] Connected to Jetson Orin Nano telemetry stream.");
        const badge = document.getElementById("connectionStatus");
        if (badge) badge.textContent = "CONNECTED (15W)";
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        } catch (e) {
            console.error("Error parsing WS packet:", e);
        }
    };

    ws.onclose = () => {
        console.log("[WS] Disconnected. Reconnecting in 2 seconds...");
        const badge = document.getElementById("connectionStatus");
        if (badge) badge.textContent = "OFFLINE (RECONNECTING)";
        setTimeout(initWebSocket, 2000);
    };

    ws.onerror = (err) => {
        console.error("[WS] Error:", err);
    };
}

// Update DOM with Telemetry Data
function updateDashboard(data) {
    // 1. Hardware & Latency Metrics (8 Metrics)
    const elPrep = document.getElementById("valPrepTime");
    const elInfer = document.getElementById("valInferTime");
    const elFps = document.getElementById("valFps");
    const elLiveFps = document.getElementById("valLiveFps");
    const elGpu = document.getElementById("valGpuLoad");
    const elCpu = document.getElementById("valCpuLoad");
    const elTemp = document.getElementById("valTemperature");
    const elVramUsed = document.getElementById("valVramUsed");
    const elVramTot = document.getElementById("valVramTotal");
    const elRamUsed = document.getElementById("valRamUsed");
    const elRamTot = document.getElementById("valRamTotal");

    if (elPrep) elPrep.textContent = Number(data.data_preprocessing_ms).toFixed(1);
    if (elInfer) elInfer.textContent = Number(data.inference_ms).toFixed(1);
    
    // FPS Metrics (Both on Tile & on Floating Video Pill)
    if (data.fps !== undefined) {
        const fpsStr = Number(data.fps).toFixed(1);
        if (elFps) elFps.textContent = fpsStr;
        if (elLiveFps) elLiveFps.textContent = fpsStr;
    }

    if (elGpu) elGpu.textContent = Math.round(data.gpu_load);
    if (elCpu) elCpu.textContent = Math.round(data.cpu_load);
    if (elTemp) elTemp.textContent = Math.round(data.temperature);
    
    // VRAM Metrics (GPU Memory used by the Model)
    const elVramUnit = document.getElementById("valVramUnit");
    if (elVramUsed) {
        if (data.vram_used_mb !== undefined) {
            const mb = Number(data.vram_used_mb);
            if (mb >= 1024) {
                elVramUsed.textContent = (mb / 1024).toFixed(2);
                if (elVramUnit) elVramUnit.textContent = "GB";
            } else {
                elVramUsed.textContent = mb.toFixed(1);
                if (elVramUnit) elVramUnit.textContent = "MB";
            }
        } else if (data.vram_used_gb !== undefined) {
            const gb = Number(data.vram_used_gb);
            if (gb < 1.0) {
                elVramUsed.textContent = (gb * 1024).toFixed(1);
                if (elVramUnit) elVramUnit.textContent = "MB";
            } else {
                elVramUsed.textContent = gb.toFixed(2);
                if (elVramUnit) elVramUnit.textContent = "GB";
            }
        }
    }
    if (elVramTot && data.vram_total_gb !== undefined) {
        elVramTot.textContent = Number(data.vram_total_gb).toFixed(1);
    }

    // System RAM Metrics
    if (elRamUsed && data.ram_used_gb !== undefined) {
        elRamUsed.textContent = Number(data.ram_used_gb).toFixed(1);
    }
    if (elRamTot && data.ram_total_gb !== undefined) {
        elRamTot.textContent = Number(data.ram_total_gb).toFixed(1);
    }

    // Hardware Device Name Badge Sync
    const connBadge = document.getElementById("connectionStatus");
    if (connBadge && data.device_name) {
        connBadge.textContent = data.device_name;
    }

    // 2. Status Highlight Text (Figure 8: STRONG / OPTIMAL)
    const statusElem = document.getElementById("statusHighlight");
    if (statusElem) {
        statusElem.textContent = data.status || "OPTIMAL";
    }

    // 3. Current Model Weight Sync
    if (data.current_weight && data.current_weight !== currentActiveWeight) {
        currentActiveWeight = data.current_weight;
        const selectEl = document.getElementById("selectModelWeight");
        if (selectEl && selectEl.value !== data.current_weight) {
            selectEl.value = data.current_weight;
        }
    }

    // 4. Source Mode & View Container Sync (Realtime vs. Offline Dual)
    if (data.source_type && data.source_type !== currentSourceType) {
        syncDisplayMode(data.source_type);
        if (data.source_type !== "camera") {
            refreshOfflineImages();
        }
    }

    // 5. Illumination Scale Segmented Pills
    if (data.scale_distribution) {
        const dist = data.scale_distribution;
        updateScalePill("pillExtremeLow", "barExtremeLow", dist["Extreme Low"] || 0.05);
        updateScalePill("pillLow", "barLow", dist["Low"] || 0.12);
        updateScalePill("pillMedium", "barMedium", dist["Medium"] || 0.28);
        updateScalePill("pillOptimal", "barOptimal", dist["Optimal"] || 0.95);

        // Highlight active highest pill
        const pills = [
            { id: "pillExtremeLow", val: dist["Extreme Low"] || 0 },
            { id: "pillLow", val: dist["Low"] || 0 },
            { id: "pillMedium", val: dist["Medium"] || 0 },
            { id: "pillOptimal", val: dist["Optimal"] || 0 }
        ];
        pills.sort((a, b) => b.val - a.val);
        
        ["pillExtremeLow", "pillLow", "pillMedium", "pillOptimal"].forEach(pid => {
            const el = document.getElementById(pid);
            if (el) el.classList.remove("active");
        });
        const highestEl = document.getElementById(pills[0].id);
        if (highestEl) highestEl.classList.add("active");
    }

    // 6. Waveform Array
    if (data.waveform && Array.isArray(data.waveform)) {
        currentWaveformData = data.waveform;
    }
}

function updateScalePill(pillId, barId, score) {
    const bar = document.getElementById(barId);
    if (bar) {
        bar.style.width = `${Math.min(100, Math.round(score * 100))}%`;
    }
}

// Fetch & Populate Available Model Weights into Header Selector
function loadAvailableWeights() {
    fetch("/api/weights")
        .then(res => res.json())
        .then(data => {
            if (data.status === "success" && data.categories) {
                const selectEl = document.getElementById("selectModelWeight");
                if (!selectEl) return;
                
                selectEl.innerHTML = "";

                data.categories.forEach(cat => {
                    const group = document.createElement("optgroup");
                    group.label = cat.category;

                    cat.items.forEach(item => {
                        const opt = document.createElement("option");
                        opt.value = item.path;
                        opt.textContent = item.name;
                        if (item.path === data.current_weight) {
                            opt.selected = true;
                        }
                        group.appendChild(opt);
                    });

                    selectEl.appendChild(group);
                });

                currentActiveWeight = data.current_weight;
                selectEl.value = data.current_weight;

                selectEl.addEventListener("change", (e) => {
                    const newWeight = e.target.value;
                    showLoading("Loading model & inferring CIDNet...");
                    selectEl.disabled = true;
                    const fd = new FormData();
                    fd.append("weight_path", newWeight);

                    fetch("/api/set_weight", {
                        method: "POST",
                        body: fd
                    })
                    .then(r => r.json())
                    .then(res => {
                        if (res.status === "success") {
                            currentActiveWeight = res.current_weight;
                            if (currentSourceType !== "camera") {
                                refreshOfflineImages();
                            } else {
                                hideLoading();
                            }
                        } else {
                            hideLoading();
                        }
                        selectEl.disabled = false;
                    })
                    .catch(err => {
                        console.error("Error setting model weight:", err);
                        hideLoading();
                        selectEl.disabled = false;
                    });
                });
            }
        })
        .catch(err => console.error("Error loading weights list:", err));
}

// Loading Spinner Helpers
function showLoading(text = "Inferring PyTorch CIDNet...") {
    const overlay = document.getElementById("loadingOverlay");
    const label = document.getElementById("loadingText");
    if (label) label.textContent = text;
    if (overlay) overlay.style.display = "flex";
}

function hideLoading() {
    const overlay = document.getElementById("loadingOverlay");
    if (overlay) overlay.style.display = "none";
}

// ==========================================
// Client Webcam Streaming Engine (WebRTC + WebSocket)
// ==========================================
async function startClientCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Trình duyệt không hỗ trợ getUserMedia hoặc bị chặn do không dùng HTTPS/localhost.");
    }

    const videoEl = document.getElementById("clientWebcamVideo");
    const canvasEl = document.getElementById("clientWebcamCanvas");
    if (!videoEl || !canvasEl) {
        throw new Error("Không tìm thấy phần tử video/canvas để thu webcam.");
    }

    // Yêu cầu camera client, tự động cấu hình kích thước 600x400
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 600 },
            height: { ideal: 400 },
            frameRate: { ideal: 30, max: 30 }
        },
        audio: false
    });

    clientMediaStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();

    // Cài đặt kích thước canvas 256x256
    canvasEl.width = 256;
    canvasEl.height = 256;

    // Kết nối WebSocket chuyên dụng gửi nhận frame client
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/client_camera`;

    return new Promise((resolve, reject) => {
        clientCameraWs = new WebSocket(wsUrl);
        clientCameraWs.binaryType = "blob";

        clientCameraWs.onopen = () => {
            isClientCameraActive = true;
            clientFpsTracker.lastTime = performance.now();
            clientFpsTracker.frames = 0;
            sendClientWebcamFrame();
            resolve();
        };

        clientCameraWs.onmessage = (event) => {
            const blob = event.data;
            if (blob instanceof Blob) {
                const url = URL.createObjectURL(blob);
                const liveImg = document.getElementById("liveStream");
                if (liveImg) {
                    const oldUrl = liveImg.src;
                    liveImg.src = url;
                    if (oldUrl && oldUrl.startsWith("blob:")) {
                        URL.revokeObjectURL(oldUrl);
                    }
                }

                // Cập nhật FPS đo từ client
                clientFpsTracker.frames++;
                const now = performance.now();
                if (now - clientFpsTracker.lastTime >= 1000) {
                    clientFpsTracker.fps = (clientFpsTracker.frames * 1000 / (now - clientFpsTracker.lastTime));
                    clientFpsTracker.frames = 0;
                    clientFpsTracker.lastTime = now;
                    const fpsEl = document.getElementById("valLiveFps");
                    if (fpsEl) fpsEl.textContent = clientFpsTracker.fps.toFixed(1);
                }
            }
            clientFrameSending = false;
            // Gửi frame kế tiếp sau khi đã nhận xong frame trước (tối ưu hóa băng thông & độ trễ)
            if (isClientCameraActive) {
                requestAnimationFrame(sendClientWebcamFrame);
            }
        };

        clientCameraWs.onerror = (err) => {
            console.error("Client Camera WebSocket error:", err);
            reject(err);
        };

        clientCameraWs.onclose = () => {
            isClientCameraActive = false;
        };
    });
}

function sendClientWebcamFrame() {
    if (!isClientCameraActive || clientFrameSending) return;
    if (!clientCameraWs || clientCameraWs.readyState !== WebSocket.OPEN) return;

    const videoEl = document.getElementById("clientWebcamVideo");
    const canvasEl = document.getElementById("clientWebcamCanvas");
    if (!videoEl || !canvasEl) return;
    if (videoEl.videoWidth === 0 || videoEl.videoHeight === 0) {
        requestAnimationFrame(sendClientWebcamFrame);
        return;
    }

    const ctx = canvasEl.getContext("2d");
    // Center-crop khung hình từ webcam về hình vuông rồi co giãn về đúng 256x256
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const minDim = Math.min(vw, vh);
    const sx = (vw - minDim) / 2;
    const sy = (vh - minDim) / 2;
    ctx.drawImage(videoEl, sx, sy, minDim, minDim, 0, 0, 256, 256);

    clientFrameSending = true;
    canvasEl.toBlob((blob) => {
        if (blob && clientCameraWs && clientCameraWs.readyState === WebSocket.OPEN) {
            clientCameraWs.send(blob);
        } else {
            clientFrameSending = false;
        }
    }, "image/jpeg", 0.85);
}

function stopClientCamera() {
    isClientCameraActive = false;
    clientFrameSending = false;

    if (clientCameraWs) {
        clientCameraWs.close();
        clientCameraWs = null;
    }

    if (clientMediaStream) {
        clientMediaStream.getTracks().forEach(track => track.stop());
        clientMediaStream = null;
    }

    const videoEl = document.getElementById("clientWebcamVideo");
    if (videoEl) {
        videoEl.srcObject = null;
    }

    const liveImg = document.getElementById("liveStream");
    if (liveImg && liveImg.src && liveImg.src.startsWith("blob:")) {
        URL.revokeObjectURL(liveImg.src);
        liveImg.src = "";
    }
}

// Display Mode Switcher (Realtime Single Output vs. Offline Dual Input/Output)
function syncDisplayMode(sourceType) {
    currentSourceType = sourceType;
    const isCamera = (sourceType === "camera" || sourceType === "client_camera");
    const realtimeBox = document.getElementById("realtimeContainer");
    const offlineBox = document.getElementById("offlineDualContainer");
    const modeBadge = document.getElementById("activeModeBadge");
    const btnCamera = document.getElementById("btnToggleCamera");
    const camText = document.getElementById("camStatusText");
    const liveStream = document.getElementById("liveStream");

    if (isCamera) {
        hideLoading();
        // Chế độ REALTIME: Chỉ hiển thị duy nhất Output đầu ra
        if (realtimeBox) realtimeBox.style.display = "flex";
        if (offlineBox) offlineBox.style.display = "none";
        if (modeBadge) {
            modeBadge.textContent = "CLIENT CAM (256x256)";
            modeBadge.className = "mode-status-badge realtime";
        }
        if (btnCamera) btnCamera.classList.add("active");
        if (camText) camText.textContent = "Cam Active";

        // Nếu là host camera (không phải client camera), nạp stream MJPEG từ server
        if (sourceType === "camera" && !isClientCameraActive) {
            if (liveStream && (!liveStream.src || !liveStream.src.includes("/video_feed"))) {
                liveStream.src = `/video_feed?t=${Date.now()}`;
            }
        }
    } else {
        // Chế độ OFFLINE (1 ảnh): Hiển thị cả Input (trái) và Output (phải)
        if (isClientCameraActive) {
            stopClientCamera();
        }
        if (realtimeBox) realtimeBox.style.display = "none";
        if (offlineBox) offlineBox.style.display = "grid";
        if (modeBadge) {
            modeBadge.textContent = "OFFLINE DUAL";
            modeBadge.className = "mode-status-badge offline";
        }
        if (btnCamera) btnCamera.classList.remove("active");
        if (camText) camText.textContent = "Client Cam";

        // Tắt feed khi xem ảnh tĩnh để tiết kiệm tối đa tài nguyên
        if (liveStream) {
            liveStream.src = "";
        }
    }
}

// Tải lại cả Input và Output ảnh tĩnh với timestamp cache buster
function refreshOfflineImages() {
    const inputImg = document.getElementById("offlineInputImg");
    const outputImg = document.getElementById("offlineOutputImg");
    const ts = Date.now();
    
    let pending = 0;
    const onImgFinished = () => {
        pending--;
        if (pending <= 0) hideLoading();
    };

    if (inputImg) {
        pending++;
        inputImg.onload = onImgFinished;
        inputImg.onerror = onImgFinished;
        inputImg.src = `/offline_input?t=${ts}`;
    }
    if (outputImg) {
        pending++;
        outputImg.onload = onImgFinished;
        outputImg.onerror = onImgFinished;
        outputImg.src = `/offline_output?t=${ts}`;
    }
    if (pending === 0) hideLoading();
}

// Initialize Camera, Upload & Dataset Navigation Controls
function initControls() {
    // Ban đầu kích hoạt chế độ hiển thị mặc định
    syncDisplayMode("sample");
    refreshOfflineImages();

    // 1. Bật/Tắt Live Client Camera Realtime
    const btnCamera = document.getElementById("btnToggleCamera");
    if (btnCamera) {
        btnCamera.addEventListener("click", async () => {
            const isCurrentlyActive = isClientCameraActive || (currentSourceType === "client_camera" || currentSourceType === "camera");

            if (isCurrentlyActive) {
                // Tắt Client Camera, chuyển về chế độ mẫu Offline
                stopClientCamera();
                const fd = new FormData();
                fd.append("action", "sample");
                try {
                    const res = await fetch("/api/source", { method: "POST", body: fd });
                    const d = await res.json();
                    syncDisplayMode(d.source_type);
                    refreshOfflineImages();
                } catch (e) {
                    console.error("Error switching to sample:", e);
                }
            } else {
                // Bật Client Camera trực tiếp từ webcam trình duyệt
                showLoading("Đang yêu cầu quyền truy cập Camera của bạn...");
                try {
                    await startClientCamera();
                    const fd = new FormData();
                    fd.append("action", "client_camera");
                    const res = await fetch("/api/source", { method: "POST", body: fd });
                    const d = await res.json();
                    syncDisplayMode(d.source_type);
                } catch (err) {
                    console.error("Error starting client camera:", err);
                    hideLoading();
                    alert("Không thể khởi động camera của trình duyệt:\n" + (err.message || err) + 
                          "\n\nLưu ý: Nếu truy cập qua IP LAN (không phải localhost), trình duyệt yêu cầu kết nối HTTPS hoặc bật cờ chrome://flags/#unsafely-treat-insecure-origin-as-secure");
                }
            }
        });
    }

    // 2. Tải ảnh tùy chọn (Custom Low-Light Image Upload)
    const uploadInput = document.getElementById("imageUploadInput");
    if (uploadInput) {
        uploadInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                if (isClientCameraActive) {
                    stopClientCamera();
                }
                showLoading("Uploading & running CIDNet inference...");
                const fd = new FormData();
                fd.append("file", e.target.files[0]);

                fetch("/api/upload_image", {
                    method: "POST",
                    body: fd
                })
                .then(r => r.json())
                .then(res => {
                    if (res.status === "success") {
                        syncDisplayMode("sample");
                        uploadInput.value = "";
                        refreshOfflineImages();
                    } else {
                        alert("Error: " + (res.message || "Upload failed"));
                        hideLoading();
                    }
                })
                .catch(err => {
                    console.error("Error uploading image:", err);
                    hideLoading();
                });
            }
        });
    }

    // 3. LOL Dataset Sample Navigation (Next / Prev)
    const btnNext = document.getElementById("btnNextSample");
    const btnPrev = document.getElementById("btnPrevSample");

    if (btnNext) {
        btnNext.addEventListener("click", () => {
            if (isClientCameraActive) {
                stopClientCamera();
            }
            showLoading("Inferring next sample with CIDNet...");
            btnNext.disabled = true;
            btnPrev.disabled = true;
            const fd = new FormData();
            fd.append("action", "next");
            fetch("/api/source", { method: "POST", body: fd })
                .then(r => r.json())
                .then(() => {
                    syncDisplayMode("sample");
                    refreshOfflineImages();
                    btnNext.disabled = false;
                    btnPrev.disabled = false;
                })
                .catch(err => {
                    console.error("Error switching next sample:", err);
                    hideLoading();
                    btnNext.disabled = false;
                    btnPrev.disabled = false;
                });
        });
    }

    if (btnPrev) {
        btnPrev.addEventListener("click", () => {
            if (isClientCameraActive) {
                stopClientCamera();
            }
            showLoading("Inferring prev sample with CIDNet...");
            btnNext.disabled = true;
            btnPrev.disabled = true;
            const fd = new FormData();
            fd.append("action", "prev");
            fetch("/api/source", { method: "POST", body: fd })
                .then(r => r.json())
                .then(() => {
                    syncDisplayMode("sample");
                    refreshOfflineImages();
                    btnNext.disabled = false;
                    btnPrev.disabled = false;
                })
                .catch(err => {
                    console.error("Error switching prev sample:", err);
                    hideLoading();
                    btnNext.disabled = false;
                    btnPrev.disabled = false;
                });
        });
    }
}

window.addEventListener("beforeunload", () => {
    if (isClientCameraActive) {
        stopClientCamera();
    }
});
