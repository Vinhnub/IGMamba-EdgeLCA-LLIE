"""
EdgeLLIE - FastAPI Server & WebSocket Hub
Phục vụ giao diện Dashboard tăng sáng ảnh và giám sát phần cứng NVIDIA Jetson Orin Nano.
"""
import os
import json
import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import config
from .jetson_hardware import jetson_monitor
from .llie_engine import llie_engine
from .camera_stream import camera_manager
from .luminance_analyzer import luminance_analyzer

app = FastAPI(title="EdgeLLIE - Jetson Orin Nano Dashboard", version="1.4.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>EdgeLLIE Dashboard is loading...</h1>")

@app.get("/video_feed")
async def video_feed():
    """Streaming video MJPEG thời gian thực (chế độ camera: chỉ hiển thị output)."""
    return StreamingResponse(
        camera_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/offline_input")
async def get_offline_input():
    """Lấy ảnh Input (gốc thiếu sáng) cho chế độ offline."""
    img_bytes = camera_manager.get_raw_jpeg()
    return Response(content=img_bytes, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

@app.get("/offline_output")
async def get_offline_output():
    """Lấy ảnh Output (đã tăng sáng) cho chế độ offline."""
    img_bytes = camera_manager.get_enhanced_jpeg()
    return Response(content=img_bytes, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    WebSocket đẩy telemetry phần cứng (GPU, CPU, Temp, RAM, VRAM, FPS),
    model weights và dữ liệu đồ thị sóng thời gian thực.
    Tần số: 10Hz (100ms một lần) cho hiển thị mượt mà.
    """
    await websocket.accept()
    try:
        while True:
            # 1. Đọc thông số phần cứng từ Jetson Orin Nano
            hw_metrics = jetson_monitor.get_metrics()
            
            # 2. Lấy thông số từ engine tăng sáng & camera
            prep_ms = camera_manager.latest_telemetry.get("prep_time_ms", llie_engine.last_prep_time)
            infer_ms = camera_manager.latest_telemetry.get("infer_time_ms", llie_engine.last_infer_time)
            fps_val = camera_manager.latest_telemetry.get("fps", camera_manager.current_fps)
            status = camera_manager.latest_telemetry.get("status", "OPTIMAL")
            scale_dist = camera_manager.latest_telemetry.get("scale_distribution", llie_engine.scale_distribution)
            curr_weight = camera_manager.latest_telemetry.get("current_weight", llie_engine.current_weight)
            
            # 3. Lấy dữ liệu đồ thị sóng Luminance Profile
            waveform_data = luminance_analyzer.get_waveform_points()
            
            is_cam_mode = (camera_manager.source_type == "camera")
            
            # Lấy dung lượng VRAM thực tế dùng để nạp mô hình
            model_mem = llie_engine.get_model_memory_usage()
            
            # Gom gói dữ liệu
            packet = {
                # Thông số phần cứng deploy Jetson Orin Nano
                "data_preprocessing_ms": prep_ms,
                "inference_ms": infer_ms,
                "fps": fps_val,
                "gpu_load": hw_metrics["gpu_load"],
                "cpu_load": hw_metrics["cpu_load"],
                "temperature": hw_metrics["temperature"],
                "vram_used_mb": model_mem["mb"],
                "vram_used_gb": model_mem["gb"],
                "vram_total_gb": hw_metrics.get("vram_total_gb", 8.0),
                "ram_used_gb": hw_metrics["ram_used_gb"],
                "ram_total_gb": hw_metrics["ram_total_gb"],
                "is_jetson": hw_metrics["is_jetson"],
                "power_mode": hw_metrics["power_mode"],
                "device_name": hw_metrics.get("device_name", "NVIDIA Jetson Orin Nano (15W)"),
                
                # Trạng thái tăng sáng & thang đo
                "status": status,
                "scale_distribution": scale_dist,
                "current_weight": curr_weight,
                
                # Dữ liệu sóng
                "waveform": waveform_data,
                
                # Nguồn dữ liệu & chế độ hiển thị (Realtime vs Offline)
                "gamma": camera_manager.gamma,
                "alpha_s": camera_manager.alpha_s,
                "alpha_i": camera_manager.alpha_i,
                "source_type": camera_manager.source_type,
                "is_camera_active": is_cam_mode
            }
            
            await websocket.send_text(json.dumps(packet))
            await asyncio.sleep(1.0 / config.TELEMETRY_FPS)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket Error] {e}")

@app.get("/api/weights")
async def get_weights():
    """Lấy danh sách tất cả các weights có trong hệ thống và weight hiện tại."""
    return {
        "status": "success",
        "current_weight": llie_engine.current_weight,
        "categories": llie_engine.get_weights_list()
    }

@app.post("/api/set_weight")
async def set_weight(weight_path: str = Form(...)):
    """Chọn và nạp trọng số mô hình mới."""
    success = llie_engine.set_weight(weight_path)
    # Re-evaluate frame with new weight if offline
    if camera_manager.source_type != "camera":
        camera_manager.is_dirty = True
        camera_manager.read_processed_frame(force_recompute=True)
    return {
        "status": "success" if success else "error",
        "current_weight": llie_engine.current_weight
    }

@app.post("/api/source")
async def change_source(action: str = Form(...)):
    """Chuyển nguồn camera / ảnh mẫu."""
    if action == "camera":
        camera_manager.switch_source("camera")
    elif action == "sample":
        camera_manager.switch_source("sample")
    elif action == "next":
        camera_manager.next_sample()
    elif action == "prev":
        camera_manager.prev_sample()
    
    return {
        "status": "success",
        "source_type": camera_manager.source_type,
        "sample_index": camera_manager.current_sample_idx,
        "is_camera_active": (camera_manager.source_type == "camera")
    }

@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload ảnh tùy ý từ máy tính để tăng sáng trực tiếp."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            if camera_manager.cap:
                camera_manager.cap.release()
                camera_manager.cap = None
            camera_manager.cached_sample_img = img
            camera_manager.source_type = "sample"
            camera_manager.is_dirty = True
            camera_manager.read_processed_frame(force_recompute=True)
            return {"status": "success", "message": "Image loaded successfully", "source_type": "sample"}
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid image file"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("edge_llie_jetson.app:app", host=config.HOST, port=config.PORT, reload=False)
