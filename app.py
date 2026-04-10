from flask import Flask, request, jsonify, render_template
import os
import requests # Used for the OpenCage API call
from PIL import Image
from io import BytesIO
from google import genai 
import time
import re # For word counting

# --- ### NEW: YOLOv8 Setup ### ---
from ultralytics import YOLO
import numpy as np
# --- ### END NEW ### ---

# =======================================================================
# 1. API Configuration & Initialization
# =================================S======================================

os.environ['GEMINI_API_KEY'] = 'REDACTED'
OPENCAGE_API_KEY = 'REDACTED' 

try:
    client = genai.Client()
    print("Gemini client initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini client. Check your API key: {e}")
    
# --- 2. Flask Setup ---
app = Flask(__name__)

# --- NEW: Global Metric Tracker ---
call_history = {'successes': 0, 'failures': 0}

# --- 3. ### NEW: Direct ML Model Setup (YOLOv8) ### ---

# We are using 'yolov8n.pt' (Nano), the smallest and fastest version.
# The '.pt' file will be AUTOMATICALLY downloaded by the 'ultralytics'
# library on the first run. No more file hunting!
try:
    DETECTION_MODEL = YOLO('yolov8n.pt')
    print("Ultralytics YOLOv8 model loaded successfully.")
    # Run a dummy inference to make sure everything is warmed up
    DETECTION_MODEL(np.zeros((640, 640, 3)), verbose=False) 
    print("YOLOv8 model is warmed up and ready.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    DETECTION_MODEL = None

# --- ### NEW: Re-written detection function ### ---
def run_object_detection(image_bytes):
    """
    This is our "direct" ML function, now using YOLOv8.
    This code is much simpler.
    """
    if DETECTION_MODEL is None:
        print("YOLOv8 model not loaded. Skipping detection.")
        return []

    try:
        # 1. Load image from bytes
        # The ultralytics model can directly read a PIL Image
        img_pil = Image.open(BytesIO(image_bytes))

        # 2. Run the model
        # verbose=False stops it from printing tons of text to your console
        
        # --- ### THE FIX ### ---
        # We add 'conf=0.5' to only detect objects with 50% or higher confidence.
        # This will filter out the "ghost" bottles and cups.
        results = DETECTION_MODEL(img_pil, verbose=False, conf=0.5)
        # --- ### END FIX ### ---
        
        # 3. Process the results
        detected_objects = set()
        
        # Get the first result (we only sent one image)
        result = results[0]
        
        # Get the human-readable names (e.g., 'person', 'cup')
        names = result.names
        
        # Loop through all detected boxes
        # 'c' is the class number (e.g., 0 for 'person', 47 for 'cup')
        for c in result.boxes.cls:
            detected_objects.add(names[int(c)])

        return list(detected_objects)

    except Exception as e:
        print(f"Error during YOLOv8 detection: {e}")
        return []
# --- ### END NEW ### ---


def calculate_word_count(text):
    """Calculates word count using regex."""
    if not text:
        return 0
    return len(re.findall(r'\b\w+\b', text))

def get_reliability_score():
    """Calculates reliability based on call history."""
    total_calls = call_history['successes'] + call_history['failures']
    if total_calls == 0:
        return "N/A"
    
    success_rate = (call_history['successes'] / total_calls) * 100
    return f"{success_rate:.0f}% ({call_history['successes']}/{total_calls})"


# =======================================================================
# 4. Location Utility Function (OpenCage Geocoding)
# =======================================================================
# (This section is unchanged)

def get_geocoding_info(latitude, longitude, question=None):
    if not OPENCAGE_API_KEY or OPENCAGE_API_KEY == 'YOUR_OPENCAGE_API_KEY_HERE':
        call_history['failures'] += 1
        return {"narration": "Location service API key is missing. Please set your OpenCage key.", "success": False}

    start_time = time.time()
    
    try:
        url = "https://api.opencagedata.com/geocode/v1/json"
        
        if question and "nearest" in question.lower():
            poi_type = question.lower().split("nearest")[1].strip().split(' ')[0]
            return get_poi_from_ai(latitude, longitude, poi_type) 

        else:
            params = {
                'key': OPENCAGE_API_KEY,
                'q': f"{latitude},{longitude}",
                'pretty': 0,
                'no_annotations': 1,
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            narration = "I found your coordinates but could not resolve an address."
            if data['results']:
                address = data['results'][0]['formatted']
                narration = f"You are currently at {address}."
            
            call_history['successes'] += 1
            metrics = {
                'latency': time.time() - start_time,
                'wordCount': calculate_word_count(narration),
                'reliability': get_reliability_score()
            }
            
            return {"narration": narration, "metrics": metrics, "success": True}
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenCage API: {e}")
        call_history['failures'] += 1
        metrics = {
            'latency': time.time() - start_time,
            'wordCount': 0,
            'reliability': get_reliability_score()
        }
        return {"narration": "Location service is currently unavailable due to a network error.", "metrics": metrics, "success": False}


def get_poi_from_ai(latitude, longitude, poi_type):
    start_time = time.time()
    address_response = get_geocoding_info(latitude, longitude)
    address = address_response.get('narration', 'your general location')
    
    prompt = (
        f"You are an accessibility assistant. Based on the fact that the user is near '{address}', "
        f"and asked to find the '{poi_type}', provide the most relevant answer or nearby example. "
        f"Be helpful and concise. If you cannot give a real-time location, give a realistic example."
    )
    
    narration = "I cannot find the location right now."
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt]
        )
        narration = response.text
        call_history['successes'] += 1
        success = True
        
    except Exception as e:
        print(f"Error in POI AI call: {e}")
        call_history['failures'] += 1
        success = False

    metrics = {
        'latency': time.time() - start_time,
        'wordCount': calculate_word_count(narration),
        'reliability': get_reliability_score()
    }
    
    return {"narration": narration, "metrics": metrics, "success": success}

# =======================================================================
# 5. Core AI Function (Visual Analysis)
# =======================================================================
# (This section is unchanged)

def describe_scene(image_bytes, prompt_override):
    start_time = time.time()
    narration = "Sorry, I encountered an error. The AI service may be unavailable or the image format is unsupported."
    success = False
    
    try:
        img = Image.open(BytesIO(image_bytes))
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt_override, img]
        )
        narration = response.text
        call_history['successes'] += 1
        success = True
        
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        call_history['failures'] += 1
        
    metrics = {
        'latency': time.time() - start_time,
        'wordCount': calculate_word_count(narration),
        'reliability': get_reliability_score()
    }
    
    return {"narration": narration, "metrics": metrics, "success": success}


# =======================================================================
# 6. API Endpoints and Routing
# =======================================================================
# (This section is unchanged, but /fast_detect now uses the new function)

@app.route('/')
def home():
    """Renders the main web interface (index.html)."""
    return render_template('index.html')

@app.route('/location_info', methods=['POST'])
def location_info_endpoint():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    question = data.get('question')

    if not latitude or not longitude:
        start_time = time.time() 
        metrics = {
            'latency': time.time() - start_time,
            'wordCount': 0,
            'reliability': get_reliability_score()
        }
        return jsonify({"narration": "Error: Coordinates not received.", "metrics": metrics}), 400

    result = get_geocoding_info(latitude, longitude, question)
    
    if 'metrics' not in result:
        result['metrics'] = {
            'latency': 0.00,
            'wordCount': calculate_word_count(result['narration']),
            'reliability': get_reliability_score()
        }
        
    return jsonify(result)


@app.route('/narrate', methods=['POST'])
def narrate_endpoint():
    start_time = time.time()
    
    if 'image' not in request.files:
        metrics = {
            'latency': time.time() - start_time,
            'wordCount': 0,
            'reliability': get_reliability_score()
        }
        return jsonify({"narration": "No image file received.", "metrics": metrics}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    user_question = request.form.get('question') 
    
    if user_question:
        full_prompt = (
            f"TASK: Analyze the image and answer the following question concisely and directly, suitable for a blind user: '{user_question}'"
        )
    else:
        full_prompt = "Describe this scene in a simple, helpful, and concise manner, suitable for a blind user. Start immediately with the description."

    result = describe_scene(image_bytes, prompt_override=full_prompt) 
    
    return jsonify(result)

# --- ### This endpoint now uses the new, simpler YOLOv8 function ### ---
# This is line 316
@app.route('/fast_detect', methods=['POST'])
def fast_detect_endpoint():
    """
    Handles the "Tier 1" fast object detection using the local YOLOv8 model.
    """
    if 'image' not in request.files:
        return jsonify({"objects": [], "error": "No image received"}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    
    try:
        # Call our new "direct" ML function
        detected_objects = run_object_detection(image_bytes)
        print(f"YOLOv8 Fast detection results: {detected_objects}") # Debug print
        return jsonify({"objects": detected_objects})
        
    except Exception as e:
        print(f"Error in fast_detect: {e}")
        return jsonify({"objects": [], "error": str(e)}), 500
# --- ### END ### ---


# =======================================================================
# 7. Run the Application
# =======================================================================

if __name__ == '__main__':
    # NOTE: In a production environment, debug=True should be removed.
    app.run(debug=True, port=5000)