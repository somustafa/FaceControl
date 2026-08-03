from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import base64

app = Flask(__name__)

# MediaPipe üz aşkarlama modelini başladırıq
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

@app.route('/')
def index():
    """Veb səhifəni açır"""
    return render_template('index.html')

def process_frame(image_bytes):
    """
    Brauzerdən gələn kadrı qəbul edir, MediaPipe ilə üzü tapır
    və emal olunmuş şəkli geriyə qaytarır.
    """
    # Baytları OpenCV şəklinə çeviririk
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return None

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            cw = int(bboxC.width * w)
            ch = int(bboxC.height * h)

            x, y = max(0, x), max(0, y)
            cw, ch = min(w - x, cw), min(h - y, ch)

            # Üzün ətrafına çərçivə çəkirik
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected (Server Processed)", (x, max(20, y - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Emal olunmuş şəkli qaytarmaq üçün JPEG-ə yığırıq
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@app.route('/process_stream', methods=['POST'])
def process_stream():
    """Brauzerdən (JS) davamlı olaraq şəkilləri qəbul edən API endpoint-i"""
    from flask import request
    file = request.files['image']
    image_bytes = file.read()
    
    processed_jpg = process_frame(image_bytes)
    if processed_jpg:
        return Response(processed_jpg, mimetype='image/jpeg')
    return "Error", 400

if __name__ == '__main__':
    # host='0.0.0.0' sayəsində şəbəkədəki noutbuk/telefon sayta girə bilər
    print("=== SERVER İŞƏ DÜŞDÜ ===")
    app.run(host='0.0.0.0', port=5000, debug=True)