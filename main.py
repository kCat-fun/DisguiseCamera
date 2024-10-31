import cv2
import mediapipe as mp
import numpy as np

# MediaPipeのセットアップ
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 顔の上に重ねる画像を読み込み
overlay_image = cv2.imread("image.png", cv2.IMREAD_UNCHANGED)

# Webカメラのキャプチャ
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # フレームをRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔検出を実行
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 顔の主要ランドマークを取得
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]

            # 画面上の座標に変換
            h, w, _ = frame.shape
            left_eye = np.array([int(left_eye.x * w), int(left_eye.y * h)])
            right_eye = np.array([int(right_eye.x * w), int(right_eye.y * h)])
            nose_tip = np.array([int(nose_tip.x * w), int(nose_tip.y * h)])
            chin = np.array([int(chin.x * w), int(chin.y * h)])

            # 顔の幅と高さを計算
            face_width = int(np.linalg.norm(right_eye - left_eye))
            face_height = int(np.linalg.norm(nose_tip - chin))

            # 顔全体を埋めるためのスケーリングを計算
            scale = max(face_width / (overlay_image.shape[1] * 0.4), face_height / overlay_image.shape[0])
            overlay_resized = cv2.resize(overlay_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # 回転角度の計算
            dx, dy = left_eye - right_eye
            angle = 180 - np.degrees(np.arctan2(dy, dx))

            # 画像を回転
            (overlay_h, overlay_w, _) = overlay_resized.shape
            rotation_matrix = cv2.getRotationMatrix2D((overlay_w // 2, overlay_h // 2), angle, 1)
            overlay_rotated = cv2.warpAffine(overlay_resized, rotation_matrix, (overlay_w, overlay_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

            # 顔の中心に画像を配置
            center_x, center_y = (left_eye + right_eye) // 2
            y1, y2 = max(0, center_y - overlay_h // 2), min(h, center_y + overlay_h // 2)
            x1, x2 = max(0, center_x - overlay_w // 2), min(w, center_x + overlay_w // 2)

            # フレームと画像が重なる部分のサイズ計算
            overlay_part = overlay_rotated[0:y2-y1, 0:x2-x1]

            # 画像の重ね合わせ
            alpha_overlay = overlay_part[:, :, 3] / 255.0  # アルファチャンネルを正規化
            alpha_background = 1.0 - alpha_overlay

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_overlay * overlay_part[:, :, c] +
                                          alpha_background * frame[y1:y2, x1:x2, c])

    # 表示
    cv2.imshow("Disguise Camera", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
