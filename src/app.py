from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
import sys
import os
import asyncio
import json
from datetime import datetime
import random

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_threat
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time Cyber Threat Detection System",
    description="Live Synthetic Traffic Monitor",
    version="1.1.0"
)

class TrafficData(BaseModel):
    src_port: int
    dst_port: int
    packet_size: int
    duration: float
    bytes_sent: int
    bytes_received: int
    flow_count: int
    avg_packet_rate: float
    protocol: str

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Real-Time Cyber Threat Detector</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #0f172a; color: #e2e8f0; }
                h1 { color: #22d3ee; }
                #log { height: 75vh; overflow-y: scroll; background: #1e2937; padding: 15px; border-radius: 8px; font-family: monospace; }
                .threat { color: #ef4444; font-weight: bold; }
                .normal { color: #22c55e; }
                button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            </style>
        </head>
        <body>
            <h1>🚨 Real-Time Cyber Threat Detection System</h1>
            <p><strong>Live Synthetic Network Traffic Monitor</strong></p>
            <button onclick="startSimulation()">▶️ Start Live Simulation</button>
            <button onclick="stopSimulation()">⏹️ Stop Simulation</button>
            <hr>
            <div id="log"></div>

            <script>
                let ws = null;
                let running = false;

                function startSimulation() {
                    if (running) return;
                    running = true;
                    
                    // Dynamic WebSocket URL (works both locally and on Render)
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = protocol + '//' + window.location.host + '/ws';
                    
                    ws = new WebSocket(wsUrl);
                    
                    ws.onmessage = function(event) {
                        const log = document.getElementById("log");
                        const data = JSON.parse(event.data);
                        const time = new Date().toLocaleTimeString();
                        
                        let colorClass = data.threat_detected ? "threat" : "normal";
                        log.innerHTML += `<p>[${time}] ${data.message} <span class="${colorClass}">${data.status}</span> (Prob: ${data.probability})</p>`;
                        log.scrollTop = log.scrollHeight;
                    };

                    ws.onclose = function() {
                        running = false;
                    };
                }

                function stopSimulation() {
                    if (ws) {
                        ws.close();
                    }
                    running = false;
                }
            </script>
        </body>
    </html>
    """

@app.post("/predict")
async def detect_threat(data: TrafficData):
    try:
        result = predict_threat(data.dict())
        return {
            "status": "success",
            "threat_detected": result["is_threat"],
            "probability": round(result["threat_probability"], 4),
            "confidence": result["confidence"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": "real-time"}

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Generate synthetic traffic
            traffic = {
                "src_port": random.randint(1000, 65535),
                "dst_port": random.choice([80, 443, 22, 3389, 53, random.randint(1, 65535)]),
                "packet_size": random.randint(40, 1500),
                "duration": round(random.uniform(0.1, 15), 2),
                "bytes_sent": random.randint(200, 150000),
                "bytes_received": random.randint(100, 100000),
                "flow_count": random.randint(1, 100),
                "avg_packet_rate": round(random.uniform(0.5, 2500), 2),
                "protocol": random.choice(["TCP", "UDP", "ICMP"])
            }

            result = predict_threat(traffic)

            message = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "threat_detected": result["is_threat"],
                "probability": round(result["threat_probability"], 4),
                "status": "🚨 THREAT DETECTED" if result["is_threat"] else "✅ Normal",
                "message": f"Flow: {traffic['src_port']} → {traffic['dst_port']} ({traffic['protocol']})"
            }

            await websocket.send_text(json.dumps(message))
            await asyncio.sleep(0.7)   # Adjust speed here

    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    print("🚀 Real-Time Cyber Threat Detection System Started")
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)