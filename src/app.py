from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import sys
import os

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import predict_threat
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time Cyber Threat Detection System",
    description="Supervised Machine Learning Powered Network Intrusion Detection",
    version="1.0.0"
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
    <h1>🚀 Real-Time Cyber Threat Detection System</h1>
    <p><strong>API is running successfully!</strong></p>
    <hr>
    <p><a href="/docs" target="_blank">📋 Go to Interactive Swagger Docs →</a></p>
    <p><a href="/health">🔍 Health Check</a></p>
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
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model": "loaded",
        "message": "Cyber Threat Detection API is running smoothly"
    }

if __name__ == "__main__":
    print("🚀 Starting Cyber Threat Detection API on http://127.0.0.1:8000")
    uvicorn.run(
        "src.app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True
    )