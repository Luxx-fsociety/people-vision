#!/bin/bash
# Запуск: ./run.sh
# При первом запуске сам создаст окружение, поставит пакеты и скачает модели (~250 МБ).
cd "$(dirname "$0")"

[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

mkdir -p models data
dl() {
  [ -s "models/$1" ] && return
  echo "Скачиваю $1 …"
  curl -L --fail --progress-bar -o "models/$1.part" "$2" && mv "models/$1.part" "models/$1"
}
HF=https://huggingface.co
dl yunet.onnx  "$HF/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
dl sface.onnx  "$HF/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
dl yolox.onnx  "$HF/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx"
dl pose.onnx   "$HF/opencv/pose_estimation_mediapipe/resolve/main/pose_estimation_mediapipe_2023mar.onnx"
dl hand.onnx   "$HF/opencv/handpose_estimation_mediapipe/resolve/main/handpose_estimation_mediapipe_2023feb.onnx"
dl clip_vision.onnx    "$HF/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx"
dl clip_text.onnx      "$HF/Xenova/clip-vit-base-patch32/resolve/main/onnx/text_model_quantized.onnx"
dl clip_tokenizer.json "$HF/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json"

exec .venv/bin/python app.py
