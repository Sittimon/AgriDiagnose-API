from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import requests
import json
import io
import base64
from typing import Dict, Any, List, Tuple, Optional
import os
import uuid
from datetime import datetime
import uvicorn
import random
import numpy as np
from openai import OpenAI
import dotenv;

dotenv.load_dotenv(dotenv.find_dotenv(),override=True);

app = FastAPI(
    title="Corn Disease Prediction API",
    description="API สำหรับการวิเคราะห์โรคใบข้าวโพดจากภาพ",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins - ในการใช้งานจริงควรระบุ domain ที่เฉพาะเจาะจง
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for model
model = None
class_names = None
class_to_idx = None
device = None
transform = None

# Custom Vision detection confidence threshold
CV_THRESHOLD = 0.95

# ------------------ Debug helpers ------------------
def _mk_debug_dir(prefix: str, enabled: bool) -> Optional[str]:
    if not enabled and os.getenv("DEBUG_SAVE_ALWAYS", "0") != "1":
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    req_id = uuid.uuid4().hex[:8]
    base = os.getenv("DEBUG_SAVE_DIR", "debug_runs")
    path = os.path.join(base, prefix, f"{ts}_{req_id}")
    os.makedirs(path, exist_ok=True)
    return path

def _save_json(path: Optional[str], name: str, data: Any):
    if not path:
        return
    try:
        with open(os.path.join(path, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _save_image(path: Optional[str], name: str, image: Image.Image):
    if not path:
        return
    try:
        image.save(os.path.join(path, name), format="JPEG", quality=95)
    except Exception:
        pass

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"
    except Exception:
        return ""

# ------------------ Bounding Box Drawing Helper ------------------
def draw_bounding_boxes_on_image(image: Image.Image, detections: List[Dict[str, Any]], 
                                detection_summary: Dict[str, Any]) -> Image.Image:
    """
    Draw bounding boxes on the original image with detection information.
    
    Args:
        image: Original PIL Image
        detections: List of detection metadata with bbox info
        detection_summary: Summary from Custom Vision detection
    
    Returns:
        PIL Image with bounding boxes drawn
    """
    # Create a copy of the image to draw on
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a font, fall back to default if not available
    try:
        # Try to use a larger font if available
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Colors for different detections
    colors = ["red", "blue", "green", "orange", "purple", "yellow", "cyan", "magenta"]
    
    for idx, detection in enumerate(detections):
        bbox = detection.get("bbox", {})
        confidence = detection.get("probability", 0.0)
        tag_name = detection.get("tagName", "Unknown")
        
        left = bbox.get("left", 0)
        top = bbox.get("top", 0)
        right = bbox.get("right", 0)
        bottom = bbox.get("bottom", 0)
        
        # Choose color for this detection
        color = colors[idx % len(colors)]
        
        # Draw bounding box
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        
        # Prepare label text
        label = f"{tag_name}: {confidence:.2%}"
        
        # Calculate text size and position
        if font:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        else:
            # Estimate text size if no font available
            text_width = len(label) * 8
            text_height = 12
        
        # Position label above the bounding box
        text_x = left
        text_y = max(0, top - text_height - 5)
        
        # Draw background rectangle for text
        draw.rectangle([text_x, text_y, text_x + text_width + 4, text_y + text_height + 4], 
                      fill=color, outline=color)
        
        # Draw text
        draw.text((text_x + 2, text_y + 2), label, fill="white", font=font)
    
    return img_with_boxes

# ------------------ Custom Vision Helper ------------------
def detect_and_crop_with_custom_vision(image: Image.Image, debug_dir: Optional[str] = None) -> Tuple[List[Tuple[Image.Image, Dict[str, Any]]], Dict[str, Any]]:
    """
    Call Azure Custom Vision Object Detection, keep boxes with probability >= CV_THRESHOLD,
    and return list of (cropped_image, meta) and a detection summary.
    """
    # Custom Vision configuration
    endpoint_url = "https://southeastasia.api.cognitive.microsoft.com/customvision/v3.0/Prediction/0c96d1d3-e022-47b0-a028-177007d20bdf/detect/iterations/Iteration2/image"
    prediction_key = os.getenv("CUSTOM_VISION_PREDICTION_KEY")

    summary: Dict[str, Any] = {
        "cv_configured": bool(endpoint_url and prediction_key),
        "threshold": CV_THRESHOLD,
        "count": 0,
        "predictions": [],
    }

    if not summary["cv_configured"]:
        return [], summary

    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        img_bytes = buf.getvalue()

        headers = {
            "Prediction-Key": prediction_key,
            "Content-Type": "application/octet-stream",
        }
        resp = requests.post(endpoint_url, headers=headers, data=img_bytes, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        _save_json(debug_dir, "cv_response.json", data)

        preds = data.get("predictions", []) or []
        preds = [p for p in preds if float(p.get("probability", 0.0)) >= CV_THRESHOLD]
        if not preds:
            return [], summary

        W, H = image.size
        crops: List[Tuple[Image.Image, Dict[str, Any]]] = []
        for p in preds:
            bb = p.get("boundingBox", {}) or {}
            prob = float(p.get("probability", 0.0))

            n_left = bb.get('left', 0)
            n_top = bb.get('top', 0)
            n_width = bb.get('width', 0)
            n_height = bb.get('height', 0)
            left = int(n_left * W)
            top = int(n_top * H)
            right = int((n_left + n_width) * W)
            bottom = int((n_top + n_height) * H)
            if n_width <= 0 or n_height <= 0:
                return None, f"boundingBox width/height เป็นศูนย์ (n_width={n_width}, n_height={n_height})"  
            # Clamp
            left = max(0, left)
            top = max(0, top)
            right = min(W, right)
            bottom = min(H, bottom)
            
            # left = max(0, int(bb.get("left", 0.0) * W))
            # top = max(0, int(bb.get("top", 0.0) * H))
            # width = int(bb.get("width", 0.0) * W)
            # height = int(bb.get("height", 0.0) * H)
            # right = min(W, left + width)
            # bottom = min(H, top + height)

            if right <= left or bottom <= top:
                summary["predictions"].append({
                    "probability": prob,
                    "bbox_raw": bb,
                    "tagName": p.get("tagName"),
                    "note": "Invalid bbox",
                })
                continue

            crop_img = image.crop((left, top, right, bottom))
            meta = {
                "probability": prob,
                "bbox": {
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                },
                "tagName": p.get("tagName"),
            }
            crops.append((crop_img, meta))
            summary["predictions"].append(meta)

        summary["count"] = len(crops)
        return crops, summary
    except Exception as e:
        summary["error"] = str(e)
        return [], summary


# ------------------ Load Model ------------------
def load_model_once(model_path: str):
    global model, class_names, class_to_idx, device, transform
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    
    num_classes = len(checkpoint['class2id'])
    original_class_names = list(checkpoint['class2id'].keys())
    class_to_idx = checkpoint['class2id']
    class_name_mapping = {
        'abnormality': "Herbicide Injury ",
        'blight': "Northern Corn Leaf Blight",
        'brownspot': "Brown Spot", 
        'curl': "Twisted Whorl",
        'healthy': "Healthy",
        'mildew': "Downy Mildew",
        'rust': "Rust",
        'smut': "Smut",
        'spot': "Leaf Sheath and Leaf Spot",
        'virus': "Virus"
    }
    
    # Map to new class names while preserving order
    class_names = []
    for original_name in original_class_names:
        new_name = class_name_mapping.get(original_name, original_name)
        class_names.append(new_name)
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Model loaded successfully on {device}")

# ------------------ Prediction Function ------------------
def predict_image(image: Image.Image) -> Dict[str, Any]:
    global model, class_names, transform, device
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # out = ""
        # with open("./image_tensor.txt",'w') as f:
        #     for i in range(image_tensor.shape[0]):
        #         for j in range(image_tensor.shape[1]):
        #             for k in range(image_tensor.shape[2]):
        #                 out += str(image_tensor[i][j][k]) + "\n"
        #     f.write(out)
        #print(image_tensor)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        all_probs = probabilities[0].cpu().numpy()
    
    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence_score),
        "all_probabilities": {class_names[i]: float(prob) for i, prob in enumerate(all_probs)}
    }

openai_client = OpenAI(
    base_url="https://chat-gpt-corn.openai.azure.com/openai/v1/",
    api_key=os.getenv('chat-gpt-api-key')
)
deployment_name = "gpt-4.1-mini"
# ------------------ GPT Call Function ------------------
def call_gpt(predicted_class: str, confidence: float) -> str:
    # url = 'https://vison-rd-prod-azopenai-scus.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview'
    # headers = {
    #     'Content-Type': 'application/json',
    #     'api-key': os.getenv("chat-gpt-api-key")
    # }

    user_prompt = (
        f"ผลการวิเคราะห์จากภาพใบข้าวโพดพบว่าเป็นโรค: {predicted_class} "
        f"โดยมีความมั่นใจ {confidence:.2%} "
        f"คุณช่วยอธิบายโรคนี้และแนะนำวิธีการดูแลรักษาให้หน่อย"
    )

    data = {
        "messages": [
            {"role": "system", "content": " Your Role\n You are a plant pathology researcher with deep expertise in diagnosing, analysing, and advising on crop diseases—especially those affecting maize (corn).\n\n Core Knowledge Available in the System\n\n| โรค (ภาษาไทย)                     | Disease (English)                   | สาเหตุ (Causal Agent)                                | ฤดูที่มักระบาดในไทย | หมายเหตุ / ลักษณะสำคัญ                 |\n| --------------------------------- | ----------------------------------- | ---------------------------------------------------- | ------------------- | -------------------------------------- |\n| โรคราน้ำค้างข้าวโพด           | Corn Downy Mildew                   | *Peronosclerospora* spp. (oomycete)                  | ฤดูฝน → ต้นฤดูหนาว  | สปอร์แพร่ทางลม – ระบาดเร็ว             |\n| โรคกาบใบเน่า                  | Sheath Rot                          | *Fusarium* spp. (fungus)                             | ปลายฤดูฝน           | เริ่มที่กาบล่างแล้วลุกลามขึ้นบน        |\n| โรคใบไหม้ใหญ่ *(ใบไหม้เหนือ)* | Northern Corn Leaf Blight           | *Exserohilum turcicum* (syn. *Setosphaeria turcica*) | ฤดูฝน ถึงต้นหนาว    | แผลรูปซิกก้าเรียวยาวสีเทา-น้ำตาล       |\n| โรคราสนิม                     | (Common) Rust                       | *Puccinia polysora* (fungus)                         | ปลายฤดูฝน           | ผงสปอร์สีน้ำตาล-ส้มบนใบ                |\n| โรคกาบและใบจุด                | Leaf Sheath & Leaf Spot             | *Epicoccum sorghinum* (fungus)                       | ปลายฤดูฝน           | จุดสีน้ำตาลดำบนกาบ/ใบ อาจทำให้แห้งกรอบ |\n| โรคใบด่าง (ไวรัส)             | Maize Mosaic Disease / Viral Mosaic | ไวรัส (ถ่ายทอดโดยแมลงปากดูด เช่น เพลี้ยจักจั่น)      | ฤดูแล้ง             | ใบมีลายเขียว-เหลืองเป็นคลื่น           |\n\n\n Diagnostic Guidelines\n\n 1. Prioritise symptom evidence: When given images or symptom descriptions, identify the disease from visible signs first.\n 2. Season data not mandatory: You can diagnose accurately from clear images alone; season is optional except for complex, overlapping symptoms (e.g., leaf wilts with multiple possible causes).\n 3. Confidence threshold: If confidence is < 80 %, request additional details (plant age, chemical history, weather, etc.).\n 4. Output: Return results in a Markdown table or JSON. Include English disease names, pathogen, likely season (if known), confidence (%), and brief notes.\n\n Sample Markdown Response\n\n \nmarkdown\n | ThaiName                | EnglishName        | Pathogen                     | LikelySeason | Confidence (%) | Notes |\n |-------------------------|--------------------|------------------------------|--------------|----------------|-------|\n | โรคราน้ำค้างข้าวโพด    | Corn Downy Mildew  | *Peronosclerospora* spp.     | Rainy–Winter | 92             | Distinct pale streaks along leaf veins |\n \n\n\n Your Tasks\n\n * Input → Analyse → Output: Provide English disease names, causal agent, season (if relevant), and concise management advice.\n * Keep explanations clear, concise, and scientifically accurate only."},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "stream": False
    }

    try:
        # response = requests.post(url, headers=headers, data=json.dumps(data))
        completion = openai_client.chat.completions.create(
            model=deployment_name,
            messages=data["messages"],
            temperature=data["temperature"],
            max_tokens=300
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการเรียก GPT: {str(e)}"

def set_seed(seed: int):
    #print("will set seed to", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42);

# ------------------ API Endpoints ------------------
@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    model_path = "best_jitter.pth"
    load_model_once(model_path)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Corn Disease Prediction API",
        "description": "API สำหรับการวิเคราะห์โรคใบข้าวโพดจากภาพ",
        "endpoints": {
            "predict": "/predict - POST รูปภาพเพื่อพยากรณ์โรค",
            "predict_with_advice": "/predict_with_advice - POST รูปภาพเพื่อพยากรณ์โรคพร้อมคำแนะนำ",
            "health": "/health - ตรวจสอบสถานะ API"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...), debug: Optional[bool] = False):
    """
    พยากรณ์โรคใบข้าวโพดจากรูปภาพ
    
    Returns:
        - predicted_class: ชื่อโรคที่พยากรณ์ได้
        - confidence: ความมั่นใจ (0-1)
        - all_probabilities: ความน่าจะเป็นของทุกคลาส
    """
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็นรูปภาพเท่านั้น")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        dbg = _mk_debug_dir("predict", bool(debug))
        _save_image(dbg, "input.jpg", image)
        
        # Predict
        result = predict_image(image)
        
        payload = {
            "success": True,
            "data": result,
            "message": "การพยากรณ์สำเร็จ"
        }
        _save_json(dbg, "response.json", payload)
        return JSONResponse(
            status_code=200,
            content=payload
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/predict_with_advice")
async def predict_disease_with_advice(file: UploadFile = File(...), debug: Optional[bool] = False):
    """
    พยากรณ์โรคใบข้าวโพดจากรูปภาพพร้อมคำแนะนำจาก AI
    
    Returns:
        - predicted_class: ชื่อโรคที่พยากรณ์ได้
        - confidence: ความมั่นใจ (0-1)
        - all_probabilities: ความน่าจะเป็นของทุกคลาส
        - expert_advice: คำแนะนำจากผู้เชี่ยวชาญ AI
    """
    
    # Check file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็นรูปภาพเท่านั้น")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        dbg = _mk_debug_dir("predict_with_advice", bool(debug))
        _save_image(dbg, "input.jpg", image)
        
        # Predict
        result = predict_image(image)
        
        # Get expert advice
        expert_advice = call_gpt(result["predicted_class"], result["confidence"])
        
        # Add advice to result
        result["expert_advice"] = expert_advice
        
        payload = {
            "success": True,
            "data": result,
            "message": "การพยากรณ์และคำแนะนำสำเร็จ"
        }
        _save_json(dbg, "response.json", payload)
        return JSONResponse(
            status_code=200,
            content=payload
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

@app.post("/predict_cv")
async def predict_via_custom_vision(file: UploadFile = File(...), debug: Optional[bool] = False):
    """
    อัปโหลดรูป → เรียก Custom Vision (threshold 0.95) → ครอปทุกกรอบ → จำแนกแต่ละครอปด้วยโมเดลในเครื่อง
    
    เงื่อนไขการส่งผลลัพธ์:
    - หากมี 1 ภาพ: ส่งกลับเสมอ
    - หากมี 2+ ภาพแต่ confidence < 0.875 ทั้งหมด: ส่งกลับเฉพาะภาพที่มี confidence สูงสุด
    - หากมี 2+ ภาพและมี confidence >= 0.875: ส่งกลับทุกภาพที่ confidence >= 0.875
    
    Returns:
        - data: ผลการจำแนกที่กรองแล้ว
        - visualization: รูปภาพต้นฉบับพร้อม bounding boxes และข้อมูลการ detect
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็นรูปภาพเท่านั้น")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        dbg = _mk_debug_dir("predict_cv", bool(debug))
        _save_image(dbg, "input.jpg", image)

        crops, detection_summary = detect_and_crop_with_custom_vision(image, debug_dir=dbg)
        
        # Save Custom Vision detection summary for debugging
        _save_json(dbg, "custom_vision_summary.json", detection_summary)
        
        all_results: List[Dict[str, Any]] = []
        for crop_img, det_meta in crops:
            cls = predict_image(crop_img)
            all_results.append({
                "detection": det_meta,
                "classification": cls,
            })
            # save each crop and its classification
            idx = len(all_results)
            _save_image(dbg, f"crop_{idx:02d}.jpg", crop_img)
            _save_json(dbg, f"crop_{idx:02d}_classification.json", cls)

        # Save all results before filtering for debugging
        _save_json(dbg, "all_results_before_filtering.json", all_results)

        # Create image with bounding boxes drawn
        detections_for_drawing = [result["detection"] for result in all_results]
        image_with_boxes = draw_bounding_boxes_on_image(image, detections_for_drawing, detection_summary)
        _save_image(dbg, "image_with_bounding_boxes.jpg", image_with_boxes)

        # Apply filtering logic based on your requirements
        filtered_results = []
        if len(all_results) == 1:
            # If only 1 crop: always return it
            filtered_results = [all_results[0]["classification"]]
        elif len(all_results) >= 2:
            # Check if any crops have confidence >= 0.875
            high_confidence_results = [
                result["classification"] for result in all_results 
                if result["classification"]["confidence"] >= 0.975
            ]
            
            if high_confidence_results:
                # If any crops have confidence >= 0.875: return all of them
                filtered_results = high_confidence_results
            else:
                # If none have confidence >= 0.875: return only the highest confidence one
                best_result = max(all_results, key=lambda x: x["classification"]["confidence"])
                filtered_results = [best_result["classification"]]

        payload = {
            "success": True,
            "data": filtered_results,
            "message": "การพยากรณ์สำเร็จ",
            "visualization": {
                "original_image_with_bounding_boxes": _image_to_base64(image_with_boxes),
                "detection_count": len(detections_for_drawing),
                "threshold_used": CV_THRESHOLD
            }
        }
        
        # Save filtering results and final payload for debugging
        _save_json(dbg, "filtered_results.json", filtered_results)
        _save_json(dbg, "final_response.json", payload)
        return JSONResponse(
            status_code=200,
            content=payload
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)
