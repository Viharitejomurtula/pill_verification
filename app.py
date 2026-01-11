from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="Pill Verification", version="1.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

print("API Ready!")

@app.get("/")
def home():
    return {
        "message": "Pill Verification App",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "GET /": "API Info",
            "GET /health": "Health Check",
            "POST /process_frames": "Process Camera Frames"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def detect_text_regions(frame):
    """Simple text detection using OpenCV"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get better text detection
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours (potential text regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count significant contours (likely text)
    text_regions = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 10000:  # Filter by size
            text_regions += 1
    
    return text_regions > 5  # If we detect multiple regions, likely has text

def detect_motion_and_person(frame):
    """Detect if there's motion/person in frame using color detection"""
    height, width = frame.shape[:2]
    
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Upper half (where face/hand would be)
    upper_half = mask[0:height//2, :]
    
    # Count skin pixels
    skin_pixels = cv2.countNonZero(upper_half)
    total_pixels = upper_half.size
    skin_ratio = skin_pixels / total_pixels
    
    person_detected = skin_ratio > 0.02
    hand_near_mouth = skin_ratio > 0.12
    confidence = min(skin_ratio * 5, 1.0)
    
    return person_detected, hand_near_mouth, round(confidence, 2)

@app.post("/process_frames")
async def process_frames(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid Image")
        
        # Simple text detection (no OCR, just detection)
        has_text = detect_text_regions(frame)
        
        # Motion/person detection
        person_detected, hand_to_mouth, confidence = detect_motion_and_person(frame)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ocr": {
                "text_detected": has_text,
                "full_text": "OCR disabled in lightweight mode",
                "medication_indicators": has_text,
                "word_count": 0
            },
            "pose": {
                "person_detected": person_detected,
                "hand_to_mouth": hand_to_mouth,
                "confidence": confidence
            },
            "summary": {
                "person_in_frame": person_detected,
                "medication_visible": has_text,
                "pill_taken": hand_to_mouth,
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error Processing: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
