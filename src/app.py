from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import sys
import os
import asyncio
import json
from datetime import datetime
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_threat
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time Cyber Threat Detection System",
    description="Live Synthetic Traffic Monitor",
    version="1.2.0"
)

METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "metrics.json")


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
                * { box-sizing: border-box; margin: 0; padding: 0; }
                body { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
                h1 { color: #22d3ee; margin-bottom: 6px; }
                .tabs { display: flex; gap: 8px; margin: 16px 0 0; }
                .tab-btn {
                    padding: 8px 20px; border: none; border-radius: 6px 6px 0 0;
                    background: #1e2937; color: #94a3b8; cursor: pointer; font-size: 14px;
                }
                .tab-btn.active { background: #0ea5e9; color: #fff; font-weight: bold; }
                .tab-content { display: none; background: #1e2937; border-radius: 0 8px 8px 8px; padding: 16px; }
                .tab-content.active { display: block; }

                /* Live feed */
                #log { height: 65vh; overflow-y: scroll; background: #0f172a; padding: 12px;
                       border-radius: 6px; font-family: monospace; font-size: 13px; }
                .threat { color: #ef4444; font-weight: bold; }
                .normal { color: #22c55e; }
                button { padding: 9px 20px; margin: 5px 5px 10px 0; font-size: 14px;
                         border: none; border-radius: 6px; cursor: pointer; }
                #startBtn { background: #0ea5e9; color: #fff; }
                #stopBtn  { background: #64748b; color: #fff; }

                /* Metrics */
                .model-card {
                    background: #0f172a; border-radius: 8px; padding: 18px;
                    margin-bottom: 18px; border: 1px solid #334155;
                }
                .model-card h3 { color: #22d3ee; margin-bottom: 12px; font-size: 16px; }
                .best-badge {
                    display: inline-block; background: #16a34a; color: #fff;
                    font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-left: 8px; vertical-align: middle;
                }
                .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
                .metric-box {
                    background: #1e2937; border-radius: 6px; padding: 12px; text-align: center;
                }
                .metric-box .label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
                .metric-box .value { font-size: 22px; font-weight: bold; color: #f1f5f9; margin-top: 4px; }
                .cm-grid {
                    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
                    max-width: 280px;
                }
                .cm-cell {
                    background: #1e2937; border-radius: 6px; padding: 10px; text-align: center;
                }
                .cm-cell .cm-label { font-size: 11px; color: #94a3b8; }
                .cm-cell .cm-val { font-size: 20px; font-weight: bold; margin-top: 2px; }
                .cm-cell.tp .cm-val { color: #22c55e; }
                .cm-cell.tn .cm-val { color: #22c55e; }
                .cm-cell.fp .cm-val { color: #f59e0b; }
                .cm-cell.fn .cm-val { color: #ef4444; }
                .cm-title { font-size: 12px; color: #94a3b8; margin-bottom: 6px; }
                #metrics-error { color: #f87171; font-size: 14px; padding: 10px 0; }
            </style>
        </head>
        <body>
            <h1>🚨 Real-Time Cyber Threat Detection System</h1>
            <p style="color:#94a3b8">Live Synthetic Network Traffic Monitor</p>

            <div class="tabs">
                <button class="tab-btn active" onclick="showTab('live')">Live Feed</button>
                <button class="tab-btn" onclick="showTab('metrics')">Model Metrics</button>
            </div>

            <!-- Live Feed Tab -->
            <div id="tab-live" class="tab-content active">
                <button id="startBtn" onclick="startSimulation()">▶ Start Live Simulation</button>
                <button id="stopBtn"  onclick="stopSimulation()">⏹ Stop Simulation</button>
                <div id="log"></div>
            </div>

            <!-- Metrics Tab -->
            <div id="tab-metrics" class="tab-content">
                <div id="metrics-content"><p style="color:#94a3b8">Loading metrics…</p></div>
            </div>

            <script>
                let ws = null, running = false;

                function showTab(name) {
                    document.querySelectorAll('.tab-btn').forEach((b, i) => {
                        b.classList.toggle('active', ['live','metrics'][i] === name);
                    });
                    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
                    document.getElementById('tab-' + name).classList.add('active');
                    if (name === 'metrics') loadMetrics();
                }

                function startSimulation() {
                    if (running) return;
                    running = true;
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
                    ws.onmessage = function(event) {
                        const log = document.getElementById("log");
                        const data = JSON.parse(event.data);
                        const cls = data.threat_detected ? "threat" : "normal";
                        log.innerHTML += `<p>[${data.timestamp}] ${data.message} — <span class="${cls}">${data.status}</span> (Prob: ${data.probability})</p>`;
                        log.scrollTop = log.scrollHeight;
                    };
                    ws.onclose = () => { running = false; };
                }

                function stopSimulation() {
                    if (ws) ws.close();
                    running = false;
                }

                async function loadMetrics() {
                    const container = document.getElementById('metrics-content');
                    try {
                        const resp = await fetch('/metrics');
                        if (!resp.ok) throw new Error('Metrics not found. Run train.py first.');
                        const data = await resp.json();

                        let html = '';
                        for (const m of data.models) {
                            const isBest = m.name === data.best_model;
                            const cm = m.confusion_matrix;
                            html += `
                            <div class="model-card">
                                <h3>${m.name}${isBest ? '<span class="best-badge">Best Model</span>' : ''}</h3>
                                <div class="metrics-grid">
                                    <div class="metric-box"><div class="label">Accuracy</div><div class="value">${(m.accuracy*100).toFixed(1)}%</div></div>
                                    <div class="metric-box"><div class="label">Precision</div><div class="value">${(m.precision*100).toFixed(1)}%</div></div>
                                    <div class="metric-box"><div class="label">Recall</div><div class="value">${(m.recall*100).toFixed(1)}%</div></div>
                                    <div class="metric-box"><div class="label">ROC AUC</div><div class="value">${(m.roc_auc*100).toFixed(1)}%</div></div>
                                </div>
                                <div class="cm-title">Confusion Matrix</div>
                                <div class="cm-grid">
                                    <div class="cm-cell tn"><div class="cm-label">True Negative</div><div class="cm-val">${cm.TN.toLocaleString()}</div></div>
                                    <div class="cm-cell fp"><div class="cm-label">False Positive</div><div class="cm-val">${cm.FP.toLocaleString()}</div></div>
                                    <div class="cm-cell fn"><div class="cm-label">False Negative</div><div class="cm-val">${cm.FN.toLocaleString()}</div></div>
                                    <div class="cm-cell tp"><div class="cm-label">True Positive</div><div class="cm-val">${cm.TP.toLocaleString()}</div></div>
                                </div>
                            </div>`;
                        }
                        container.innerHTML = html;
                    } catch (e) {
                        container.innerHTML = `<div id="metrics-error">⚠ ${e.message}</div>`;
                    }
                }
            </script>
        </body>
    </html>
    """


@app.get("/metrics")
async def get_metrics():
    if not os.path.exists(METRICS_PATH):
        return JSONResponse(status_code=404, content={"error": "metrics.json not found. Run train.py first."})
    with open(METRICS_PATH) as f:
        return JSONResponse(content=json.load(f))


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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
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
            await websocket.send_text(json.dumps({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "threat_detected": result["is_threat"],
                "probability": round(result["threat_probability"], 4),
                "status": "THREAT DETECTED" if result["is_threat"] else "Normal",
                "message": f"Flow: {traffic['src_port']} -> {traffic['dst_port']} ({traffic['protocol']})"
            }))
            await asyncio.sleep(0.7)
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    print("Real-Time Cyber Threat Detection System Started")
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
