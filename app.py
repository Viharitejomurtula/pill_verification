from fastapi import FastAPI, File, UploadFile, HTTPException   #File+UploadFile are used for getting frames from the frontend
from fastapi.middleware.cors import CORSMiddleware   #Cross-Origin-Resource-Sharing, allows requests from websites
import cv2
import numpy as np
import easyocr
import mediapipe as mp
from datetime import datetime
from mediapipe import tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

app = FastAPI(title="Pill Verification", version="1.0")    #starting up the app

#add CORS to allow browser requests

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],    #allows requests from anywhere
	allow_headers=["*"],    #allows all http headers
	allow_methods=["*"],	  #allows GET, POST, and other methods
)

#initialize ML model for text extraction and gesture recognition
print("Loading ML models...")
reader = easyocr.Reader(['en'], gpu=False)    #en is for english, other languages can be supported too in the future



model_path = 'pose_landmark_lite.task'
if not os.path.exists(model_path):
	print("Downloading Model...")
	url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task'
	urllib.request.urlretrieve(url, model_path)
	print("Model Loaded")

base_options = python.BaseOptions(model_asset_path=model_path)
#0.5 confidence is a good balance
options = vision.PoseLandmarkerOptions(base_options = base_options, running_mode = vision.RunningMode.IMAGE, min_pose_detection_confidence = 0.5, min_tracking_confidence = 0.5)
pose = vision.PoseLandmarker.create_from_options(options)




print("ML models loaded")


#methods

@app.get("/")
def home():
	return{
		"message": "Pill Verification App",
		"version": "1.0",
		"status": "running",
		"endpoints": {
			"GET /" : "API Info",
			"GET /health": "Health Check",
			"POST /process_frames": "Process Camera Frames"
		}
	}

@app.get("/health")
def health():
	return {"status": "healthy", "timestamp":datetime.now().isoformat()}

@app.post("/process_frames")
async def process_frames(file: UploadFile = File(...)):    #async means that other processes can be carried out side by side with this
	                                                   #and the server doesn't wait until this process is over before moving on
	#OCR for text from pill bottlee
	#MediaPose for hand to mouth gesture recognition
	
	try:
		contents = await file.read()     #entire file is read into memory as bytes, await bc process_frames is async
		nparr = np.frombuffer(contents, np.uint8)   #each number is converted into 8 bits and placed in a numpy array
		frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   #takes nparr and converst into an RGB image tensor
		
		if frame is None:
			raise HTTPException(status_code=400, detail="Invalid Image")

	
		#OCR Processing
		ocr_results = reader.readtext(frame)      #text from the frame is stored here along with confidence ("Tylenol", 0.9)
		detected_texts = [text[1] for text in ocr_results]   #stores just the text and disgards the confidence
		full_text = " ".join(detected_texts)

		#hint for words to look out for
		medication_keywords = [
			"mg", "tablet", "capsule", "pill", "daily", "take",
            "prescription", "rx", "medication", "dose", "tylenol",
            "advil", "aspirin", "ibuprofen"
		]
		
		#check if any keywords were found
		had_medication = False
		for keyword in medication_keywords:
			if keyword.lower() in full_text.lower():
				had_medication = True
				break

		#Pose Detection
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
		pose_results = pose.detect(mp_image)
		
		#initializing hand_to_mouth, confidence, and pose_detected to the opposite of what we want it to be for now
		hand_to_mouth =  False    #boolean
		confidence = 0.0          #float
		pose_detected = False     #boolean

		if pose_results.pose_landmarks:     #if no human is detected pose_landmarks is None, else 33 landmarks will be in there
			pose_detected = True
			landmarks = pose_results.pose_landmarks[0]   #list of 33 normalized points
			
			#we pull out right wrist, left wrist, and mouth as specific parts we will look at
			right_wrist =  landmarks[16]
			left_wrist = landmarks[15]
			mouth_right = landmarks[10]    #since MOUTH_CENTER doesn't exist, we average
			mouth_left = landmarks[9]      #MOUTH_RIGHT AND MOUTH_LEFT
			
			mouth_x = (mouth_right.x + mouth_left.x) / 2
			mouth_y = (mouth_right.y + mouth_left.y) / 2

			#Distance Calculation
			right_dist = np.sqrt((right_wrist.x - mouth_x)**2 + (right_wrist.y - mouth_y)**2)
			left_dist = np.sqrt((left_wrist.x - mouth_x)**2 + (left_wrist.y - mouth_y)**2)
		
			threshold = 0.15
			min_distance = min(right_dist, left_dist)
			if min_distance < threshold:
				hand_to_mouth = True
				confidence = round(1.0 - min_distance, 2)

	
		return {
			"timestamp": datetime.now().isoformat(),
			"ocr": {
				"text_detected":len(detected_texts) > 0,
				"full_text": full_text,
				"medication_indicators": had_medication,
				"word_count": len(detected_texts)
			},	
			"pose": {
				"person_detected": pose_detected,
				"hand_to_mouth": hand_to_mouth,
				"confidence": confidence
			},
			"summary": {
				"person_in_frame": pose_detected,
				"medication_visible": had_medication,
				"pill_taken": hand_to_mouth,
			}
		}

		

	except Exception as e:
		raise HTTPException(status_code=500, detail = f"Error Processing: {str(e)}")





#server startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # ← Read PORT from environment
    uvicorn.run(app, host="0.0.0.0", port=port)
