import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, request
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

app = Flask(__name__)

# MediaPipe üz aşkarlama modelini başladırıq
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)


# ==========================================
# 1. DATASET HAZIRLANMASI (Köhnə funksiyanız)
# ==========================================
def load_image_any_format(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.heic':
        pil_img = Image.open(image_path).convert('RGB')
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(image_path)

def prepare_dataset(raw_dir="data/raw", processed_dir="data/processed", target_size=(112, 112)):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic')

    if not os.path.exists(raw_dir):
        print(f"[!] '{raw_dir}' qovluğu tapılmadı. Dataset hazırlığı ötürüldü.")
        return

    print("=== DATASET HAZIRLANMASI BAŞLANDI ===")
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        for person_name in os.listdir(raw_dir):
            person_raw_path = os.path.join(raw_dir, person_name)
            if not os.path.isdir(person_raw_path):
                continue

            person_processed_path = os.path.join(processed_dir, person_name)
            os.makedirs(person_processed_path, exist_ok=True)
            count = 0

            for filename in os.listdir(person_raw_path):
                if not filename.lower().endswith(supported_extensions):
                    continue

                img_path = os.path.join(person_raw_path, filename)
                try:
                    image = load_image_any_format(img_path)
                except Exception:
                    continue

                if image is None:
                    continue

                h, w, _ = image.shape
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = fd.process(image_rgb)

                if results.detections:
                    for i, detection in enumerate(results.detections):
                        bboxC = detection.location_data.relative_bounding_box
                        x, y = max(0, int(bboxC.xmin * w)), max(0, int(bboxC.ymin * h))
                        cw, ch = min(w - x, int(bboxC.width * w)), min(h - y, int(bboxC.height * h))

                        if cw > 10 and ch > 10:
                            face_crop = image[y:y+ch, x:x+cw]
                            face_resized = cv2.resize(face_crop, target_size)
                            save_path = os.path.join(person_processed_path, f"{os.path.splitext(filename)[0]}_crop_{i+1}.jpg")
                            cv2.imwrite(save_path, face_resized)
                            count += 1
            print(f" [✔] {person_name}: {count} üz saxlanıldı.")


# ==========================================
# 2. VEB SERVER VƏ REAL-TIME ÜZ AŞKARLAMA
# ==========================================
@app.route('/')
def index():
    """Brauzerdə veb interfeysi açır"""
    return render_template('index.html')

@app.route('/process_stream', methods=['POST'])
def process_stream():
    """Noutbuk brauzerindən gələn kadrları emal edir"""
    if 'image' not in request.files:
        return "No image sent", 400

    file = request.files['image']
    image_bytes = file.read()

    # Göndərilən baytları OpenCV şəklinə çeviririk
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return "Invalid image", 400

    # MediaPipe ilə üzləri tapırıq
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = max(0, int(bboxC.xmin * w))
            y = max(0, int(bboxC.ymin * h))
            cw = min(w - x, int(bboxC.width * w))
            ch = min(h - y, int(bboxC.height * h))

            # Üzün ətrafına çərçivə çəkirik
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected (PC Server)", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Emal olunmuş kadrı JPEG formatında brauzerə geri göndəririk
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(buffer.tobytes(), mimetype='image/jpeg')


if __name__ == "__main__":
    # 1. İstəsəniz dataseti hazırlayır (Mövcud raw papkası varsa)
    prepare_dataset()

    # 2. Veb serveri başladırıq (host='0.0.0.0' şəbəkədən qoşulmağa icazə verir)
    print("\n=== FLASK SERVER İŞƏ DÜŞDÜ ===")
    print("Noutbukdan qoşulmaq üçün brauzerə daxil edin: http://<PC_IP_UNVANI>:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)