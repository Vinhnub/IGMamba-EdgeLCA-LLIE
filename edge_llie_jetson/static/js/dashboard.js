/**
 * EdgeLLIE - Dashboard Real-Time Telemetry & Input Controller
 * Giao diện tối ưu hóa theo mô hình:
 * Top Controls (Theme, Cam Client Realtime, Upload Image, Weight Model Select)
 * Middle (Low Light Input, Enhace Output)
 * Bottom (Time Inference, NIQE Output, CPU Load, Ram, Device Temparature, FPS)
 */

// Global State
let ws = null;
let currentActiveWeight = "";
let currentSourceType = "sample";

// Client Camera Webcam State
let clientMediaStream = null;
let clientCameraWs = null;
let isClientCameraActive = false;
let clientFrameSending = false;
let clientFpsTracker = { lastTime: performance.now(), frames: 0, fps: 30.0 };
let currentFacingMode = "environment"; // Mặc định camera sau cho thiết bị di động

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initWebSocket();
    initControls();
    loadAvailableWeights();
});

// ==========================================
// 1. Theme Management (Dark / Light Theme)
// ==========================================
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

    if (theme === "light") {
        body.classList.remove("dark-theme");
        body.classList.add("light-theme");
        if (themeIcon) themeIcon.textContent = "☀️";
    } else {
        body.classList.remove("light-theme");
        body.classList.add("dark-theme");
        if (themeIcon) themeIcon.textContent = "🌙";
    }
}

// ==========================================
// 2. WebSocket Telemetry (Hardware & Model Stats)
// ==========================================
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

// Update DOM with Telemetry Data (6 Cards + Header)
function updateDashboard(data) {
    // 1. Time Inference & Preprocessing
    const elInfer = document.getElementById("valInferTime");
    const elPrep = document.getElementById("valPrepTime");
    if (elInfer && data.inference_ms !== undefined) {
        elInfer.textContent = Number(data.inference_ms).toFixed(1);
    }
    if (elPrep && data.data_preprocessing_ms !== undefined) {
        elPrep.textContent = Number(data.data_preprocessing_ms).toFixed(1);
    }

    // 2. NIQE Output
    const elNiqe = document.getElementById("valNiqe");
    if (elNiqe) {
        const niqeVal = data.niqe_output !== undefined ? data.niqe_output : data.niqe_score;
        if (niqeVal !== undefined && niqeVal !== null) {
            elNiqe.textContent = Number(niqeVal).toFixed(2);
        }
    }

    // 3. CPU Load
    const elCpu = document.getElementById("valCpuLoad");
    const barCpu = document.getElementById("barCpuLoad");
    if (data.cpu_load !== undefined) {
        const cpuRound = Math.round(data.cpu_load);
        if (elCpu) elCpu.textContent = cpuRound;
        if (barCpu) barCpu.style.width = `${Math.min(100, Math.max(0, cpuRound))}%`;
    }

    // 4. Ram (System Memory)
    const elRamUsed = document.getElementById("valRamUsed");
    const elRamTot = document.getElementById("valRamTotal");
    const barRam = document.getElementById("barRamLoad");
    if (data.ram_used_gb !== undefined) {
        const ramUsed = Number(data.ram_used_gb).toFixed(1);
        if (elRamUsed) elRamUsed.textContent = ramUsed;
        const ramTot = data.ram_total_gb ? Number(data.ram_total_gb) : 8.0;
        if (elRamTot) elRamTot.textContent = ramTot.toFixed(1);
        if (barRam && ramTot > 0) {
            const pct = Math.round((Number(data.ram_used_gb) / ramTot) * 100);
            barRam.style.width = `${Math.min(100, Math.max(0, pct))}%`;
        }
    }

    // 5. Device Temperature
    const elTemp = document.getElementById("valTemperature");
    const statusTemp = document.getElementById("statusTemp");
    if (data.temperature !== undefined) {
        const tempRound = Math.round(data.temperature);
        if (elTemp) elTemp.textContent = tempRound;
        if (statusTemp) {
            if (tempRound >= 70) {
                statusTemp.textContent = "High Thermal";
                statusTemp.className = "sub-info status-warm";
            } else {
                statusTemp.textContent = "Normal Thermal";
                statusTemp.className = "sub-info status-normal";
            }
        }
    }

    // 6. FPS
    const elFps = document.getElementById("valFps");
    if (elFps && data.fps !== undefined) {
        elFps.textContent = Number(data.fps).toFixed(1);
    }

    // Hardware Device Name Badge Sync
    const connBadge = document.getElementById("connectionStatus");
    if (connBadge && data.device_name) {
        connBadge.textContent = data.device_name;
    }

    // Current Model Weight Sync
    if (data.current_weight && data.current_weight !== currentActiveWeight) {
        currentActiveWeight = data.current_weight;
        const selectEl = document.getElementById("selectModelWeight");
        if (selectEl && selectEl.value !== data.current_weight) {
            selectEl.value = data.current_weight;
        }
    }

    // Source Mode Sync (Realtime vs Offline)
    if (data.source_type && data.source_type !== currentSourceType) {
        syncDisplayMode(data.source_type);
        if (data.source_type !== "camera" && data.source_type !== "client_camera") {
            refreshOfflineImages();
        }
    }
}

// ==========================================
// 3. Pretrained Model Weights Selection
// ==========================================
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
                    showLoading("Đang nạp trọng số mô hình mới...");
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
                            if (currentSourceType !== "camera" && currentSourceType !== "client_camera") {
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
function showLoading(text = "Inferring model...") {
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
// 4. Client Webcam Realtime Streaming Engine
// ==========================================
function updateFlipButtonText() {
    const flipText = document.getElementById("camFlipText");
    if (flipText) {
        flipText.textContent = (currentFacingMode === "environment") ? "Cam: Sau" : "Cam: Trước";
    }
}

async function switchCameraFacingMode() {
    if (!isClientCameraActive) return;

    currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
    updateFlipButtonText();

    if (clientMediaStream) {
        clientMediaStream.getTracks().forEach(t => t.stop());
        clientMediaStream = null;
    }

    const videoEl = document.getElementById("clientWebcamVideo");
    if (videoEl) videoEl.srcObject = null;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 600 },
                height: { ideal: 400 },
                frameRate: { ideal: 30, max: 30 }
            },
            audio: false
        });
        clientMediaStream = stream;
        if (videoEl) {
            videoEl.srcObject = stream;
            await videoEl.play();
        }
    } catch (err) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            clientMediaStream = stream;
            if (videoEl) {
                videoEl.srcObject = stream;
                await videoEl.play();
            }
        } catch (err2) {
            console.error("Không thể đổi camera:", err2);
            alert("Không thể chuyển camera: " + (err2.message || err2.name));
        }
    }
}

async function startClientCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Trình duyệt không hỗ trợ getUserMedia hoặc bị chặn do không dùng HTTPS/localhost.");
    }

    const videoEl = document.getElementById("clientWebcamVideo");
    const canvasEl = document.getElementById("clientWebcamCanvas");
    if (!videoEl || !canvasEl) {
        throw new Error("Không tìm thấy phần tử video/canvas.");
    }

    let stream = null;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 600 },
                height: { ideal: 400 },
                frameRate: { ideal: 30, max: 30 }
            },
            audio: false
        });
    } catch (e) {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }

    clientMediaStream = stream;
    videoEl.srcObject = stream;
    await videoEl.play();
    updateFlipButtonText();

    canvasEl.width = 256;
    canvasEl.height = 256;

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
                const liveOutput = document.getElementById("liveStreamOutput");
                if (liveOutput) {
                    const oldUrl = liveOutput.src;
                    liveOutput.src = url;
                    if (oldUrl && oldUrl.startsWith("blob:")) {
                        URL.revokeObjectURL(oldUrl);
                    }
                }

                clientFpsTracker.frames++;
                const now = performance.now();
                if (now - clientFpsTracker.lastTime >= 1000) {
                    clientFpsTracker.fps = (clientFpsTracker.frames * 1000 / (now - clientFpsTracker.lastTime));
                    clientFpsTracker.frames = 0;
                    clientFpsTracker.lastTime = now;
                    const fpsEl = document.getElementById("valFps");
                    if (fpsEl) fpsEl.textContent = clientFpsTracker.fps.toFixed(1);
                }
            }
            clientFrameSending = false;
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
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;
    const minDim = Math.min(vw, vh);
    const sx = (vw - minDim) / 2;
    const sy = (vh - minDim) / 2;
    ctx.drawImage(videoEl, sx, sy, minDim, minDim, 0, 0, 256, 256);

    clientFrameSending = true;
    canvasEl.toBlob((blob) => {
        if (!blob) {
            clientFrameSending = false;
            return;
        }

        // Cập nhật khung hình Low Light Input bên trái
        const liveInput = document.getElementById("liveStreamInput");
        if (liveInput) {
            const inputUrl = URL.createObjectURL(blob);
            const oldInputUrl = liveInput.src;
            liveInput.src = inputUrl;
            if (oldInputUrl && oldInputUrl.startsWith("blob:")) {
                URL.revokeObjectURL(oldInputUrl);
            }
        }

        if (clientCameraWs && clientCameraWs.readyState === WebSocket.OPEN) {
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

    const liveInput = document.getElementById("liveStreamInput");
    if (liveInput) {
        if (liveInput.src && liveInput.src.startsWith("blob:")) {
            URL.revokeObjectURL(liveInput.src);
        }
        liveInput.src = "";
    }

    const liveOutput = document.getElementById("liveStreamOutput");
    if (liveOutput) {
        if (liveOutput.src && liveOutput.src.startsWith("blob:")) {
            URL.revokeObjectURL(liveOutput.src);
        }
        liveOutput.src = "";
    }
}

// ==========================================
// 5. Display Mode Switcher (Realtime vs Offline)
// ==========================================
function syncDisplayMode(sourceType) {
    currentSourceType = sourceType;
    const isCamera = (sourceType === "camera" || sourceType === "client_camera");

    const offlineInput = document.getElementById("offlineInputImg");
    const offlineOutput = document.getElementById("offlineOutputImg");
    const liveInput = document.getElementById("liveStreamInput");
    const liveOutput = document.getElementById("liveStreamOutput");

    const inputBadge = document.getElementById("inputModeBadge");
    const outputBadge = document.getElementById("outputModeBadge");

    const btnCamera = document.getElementById("btnToggleCamera");
    const camText = document.getElementById("camStatusText");
    const btnFlip = document.getElementById("btnFlipCamera");

    if (isCamera) {
        hideLoading();
        const plInput = document.getElementById("placeholderInput");
        const plOutput = document.getElementById("placeholderOutput");
        if (plInput) plInput.style.display = "none";
        if (plOutput) plOutput.style.display = "none";

        if (offlineInput) offlineInput.style.display = "none";
        if (offlineOutput) offlineOutput.style.display = "none";
        if (liveInput) liveInput.style.display = "block";
        if (liveOutput) liveOutput.style.display = "block";

        if (inputBadge) {
            inputBadge.textContent = "LIVE INPUT";
            inputBadge.className = "panel-badge badge-input";
        }
        if (outputBadge) {
            outputBadge.textContent = "LIVE ENHANCED";
            outputBadge.className = "panel-badge badge-output";
        }

        if (btnCamera) btnCamera.classList.add("active");
        if (camText) camText.textContent = "Cam Active (Stop)";
        if (btnFlip) {
            btnFlip.style.display = "inline-flex";
            updateFlipButtonText();
        }

        // Host camera stream fallback
        if (sourceType === "camera" && !isClientCameraActive) {
            const ts = Date.now();
            if (liveInput && (!liveInput.src || !liveInput.src.includes("/video_feed_raw"))) {
                liveInput.src = `/video_feed_raw?t=${ts}`;
            }
            if (liveOutput && (!liveOutput.src || !liveOutput.src.includes("/video_feed"))) {
                liveOutput.src = `/video_feed?t=${ts}`;
            }
        }
    } else {
        // Chế độ Offline Dual Input / Enhace Output
        if (isClientCameraActive) {
            stopClientCamera();
        }
        if (offlineInput) offlineInput.style.display = "block";
        if (offlineOutput) offlineOutput.style.display = "block";
        if (liveInput) liveInput.style.display = "none";
        if (liveOutput) liveOutput.style.display = "none";

        if (inputBadge) {
            inputBadge.textContent = "RAW INPUT";
            inputBadge.className = "panel-badge badge-input";
        }
        if (outputBadge) {
            outputBadge.textContent = "ENHANCED";
            outputBadge.className = "panel-badge badge-output";
        }

        if (btnCamera) btnCamera.classList.remove("active");
        if (camText) camText.textContent = "Cam Client Realtime";
        if (btnFlip) btnFlip.style.display = "none";

        // Xóa blob URL camera
        if (liveInput) {
            if (liveInput.src && liveInput.src.startsWith("blob:")) {
                URL.revokeObjectURL(liveInput.src);
            }
            liveInput.src = "";
        }
        if (liveOutput) {
            if (liveOutput.src && liveOutput.src.startsWith("blob:")) {
                URL.revokeObjectURL(liveOutput.src);
            }
            liveOutput.src = "";
        }
    }
}

// Tải lại ảnh tĩnh với timestamp cache buster
function refreshOfflineImages() {
    const inputImg = document.getElementById("offlineInputImg");
    const outputImg = document.getElementById("offlineOutputImg");
    const plInput = document.getElementById("placeholderInput");
    const plOutput = document.getElementById("placeholderOutput");
    const ts = Date.now();

    let pending = 0;
    const onImgFinished = () => {
        pending--;
        if (pending <= 0) hideLoading();
    };

    if (inputImg) {
        pending++;
        inputImg.onload = () => {
            if (plInput) plInput.style.display = "none";
            inputImg.style.display = "block";
            onImgFinished();
        };
        inputImg.onerror = () => {
            if (plInput) plInput.style.display = "flex";
            inputImg.style.display = "none";
            onImgFinished();
        };
        inputImg.src = `/offline_input?t=${ts}`;
    }
    if (outputImg) {
        pending++;
        outputImg.onload = () => {
            if (plOutput) plOutput.style.display = "none";
            outputImg.style.display = "block";
            onImgFinished();
        };
        outputImg.onerror = () => {
            if (plOutput) plOutput.style.display = "flex";
            outputImg.style.display = "none";
            onImgFinished();
        };
        outputImg.src = `/offline_output?t=${ts}`;
    }
    if (pending === 0) hideLoading();
}

// ==========================================
// 6. Init Controls (Buttons & File Upload)
// ==========================================
function initControls() {
    syncDisplayMode("sample");
    refreshOfflineImages();

    // 1. Cam Client Realtime Toggle Button
    const btnCamera = document.getElementById("btnToggleCamera");
    if (btnCamera) {
        btnCamera.addEventListener("click", async () => {
            const isCurrentlyActive = isClientCameraActive || (currentSourceType === "client_camera" || currentSourceType === "camera");

            if (isCurrentlyActive) {
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
                showLoading("Đang kết nối Webcam của bạn...");
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
                    alert("Không thể khởi động camera:\n" + (err.message || err) +
                          "\n\nLưu ý: Yêu cầu truy cập qua localhost hoặc kết nối an toàn HTTPS.");
                }
            }
        });
    }

    // 2. Mobile Flip Camera
    const btnFlip = document.getElementById("btnFlipCamera");
    if (btnFlip) {
        btnFlip.addEventListener("click", async () => {
            await switchCameraFacingMode();
        });
    }

    // 3. Upload Image Button
    const uploadInput = document.getElementById("imageUploadInput");
    if (uploadInput) {
        uploadInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                if (isClientCameraActive) {
                    stopClientCamera();
                }
                showLoading("Đang tải ảnh và chạy mô hình tăng sáng...");
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

    // 4. LOL Dataset Sample Navigation (Next / Prev)
    const btnNext = document.getElementById("btnNextSample");
    const btnPrev = document.getElementById("btnPrevSample");

    if (btnNext) {
        btnNext.addEventListener("click", () => {
            if (isClientCameraActive) stopClientCamera();
            showLoading("Đang tăng sáng ảnh kế tiếp...");
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
                    console.error("Error next sample:", err);
                    hideLoading();
                    btnNext.disabled = false;
                    btnPrev.disabled = false;
                });
        });
    }

    if (btnPrev) {
        btnPrev.addEventListener("click", () => {
            if (isClientCameraActive) stopClientCamera();
            showLoading("Đang tăng sáng ảnh trước...");
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
                    console.error("Error prev sample:", err);
                    hideLoading();
                    btnNext.disabled = false;
                    btnPrev.disabled = false;
                });
        });
    }

    // Phím tắt bàn phím tiện ích (Mũi tên Trái / Phải để duyệt ảnh mẫu nhanh)
    window.addEventListener("keydown", (e) => {
        if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
        if (e.key === "ArrowRight" && btnNext && !btnNext.disabled) {
            btnNext.click();
        } else if (e.key === "ArrowLeft" && btnPrev && !btnPrev.disabled) {
            btnPrev.click();
        }
    });
}

window.addEventListener("beforeunload", () => {
    if (isClientCameraActive) {
        stopClientCamera();
    }
});
