import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, ttk
import time
from PIL import Image, ImageFont, ImageDraw

# Mediapipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

# 각도 피드백 함수
def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def provide_feedback(angles_left, angles_right, target_angles):
    feedback = []
    body_parts = ["어깨", "허리", "발목"]
    for i in range(3):
        if angles_left[i] < target_angles[i] - 10:
            feedback.append(f"왼쪽 {body_parts[i]}: 더 돌려야 합니다.")
        elif angles_left[i] > target_angles[i] + 10:
            feedback.append(f"왼쪽 {body_parts[i]}: 너무 많이 돌려졌습니다.")
        else:
            feedback.append(f"왼쪽 {body_parts[i]}: 올바른 자세입니다.")

        if angles_right[i] < target_angles[i] - 10:
            feedback.append(f"오른쪽 {body_parts[i]}: 더 돌려야 합니다.")
        elif angles_right[i] > target_angles[i] + 10:
            feedback.append(f"오른쪽 {body_parts[i]}: 너무 많이 돌려졌습니다.")
        else:
            feedback.append(f"오른쪽 {body_parts[i]}: 올바른 자세입니다.")
    
    return feedback

# 목표 각도 데이터 로드
angle_data = pd.read_csv("C:/Users/kimyo/box/angles_data1.csv")
moves_angles = {
    '잽': angle_data[angle_data['Move'] == '잽'][['Shoulder-Elbow', 'Hip-Knee', 'Ankle-Foot']].mean().tolist(),
    '훅': angle_data[angle_data['Move'] == '훅'][['Shoulder-Elbow', 'Hip-Knee', 'Ankle-Foot']].mean().tolist(),
    '어퍼': angle_data[angle_data['Move'] == '어퍼'][['Shoulder-Elbow', 'Hip-Knee', 'Ankle-Foot']].mean().tolist(),
    '스트레이트': angle_data[angle_data['Move'] == '스트레이트'][['Shoulder-Elbow', 'Hip-Knee', 'Ankle-Foot']].mean().tolist()
}

# 이미지에 한글 텍스트를 추가하는 함수
def add_korean_text(image, text, position, font_path="C:/Windows/Fonts/malgun.ttf", font_size=30, color=(0, 0, 255)):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# 피드백 창을 표시하는 함수
def display_feedback(feedback):
    feedback_window = tk.Toplevel(root)
    feedback_window.title("자세 피드백")
    feedback_window.geometry("400x300")
    
    ttk.Label(feedback_window, text="자세 피드백 결과:", font=("Helvetica", 14)).pack(pady=10)
    
    for msg in feedback:
        ttk.Label(feedback_window, text=msg, font=("Helvetica", 12)).pack(anchor="w", padx=10)
    ttk.Button(feedback_window, text="확인", command=feedback_window.destroy).pack(pady=20)

# GUI 설정
def start_pose_feedback(move_name):
    target_angles = moves_angles[move_name]
    feedback = []

    cap = cv2.VideoCapture(0)
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    
    start_time = time.time()
    feedback_collected = False
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=2) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 배경 제거
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = selfie_segmentation.process(image_rgb)
            mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255
            frame = cv2.bitwise_and(frame, frame, mask=mask)

            elapsed_time = time.time() - start_time
            if elapsed_time < 3:
                frame = add_korean_text(frame, f"3초 후 {move_name}을 하세요", (50, 50))
            else:
                # 포즈 인식 시작
                results = pose.process(image_rgb)

                if results.pose_landmarks and not feedback_collected:
                    landmarks = results.pose_landmarks.landmark
                    # 필요한 관절의 좌표 추출 (왼쪽과 오른쪽 모두)
                    left_shoulder, left_elbow, left_hip, left_knee, left_ankle, left_foot = (
                        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z],
                        [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].z]
                    )

                    right_shoulder, right_elbow, right_hip, right_knee, right_ankle, right_foot = (
                        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z],
                        [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].z]
                    )

                    # 왼쪽과 오른쪽 각도 계산
                    angles_left = [
                        calculate_angle_3d(left_shoulder, left_elbow, left_foot),
                        calculate_angle_3d(left_hip, left_knee, left_elbow),
                        calculate_angle_3d(left_ankle, left_foot, left_knee)
                    ]
                    angles_right = [
                        calculate_angle_3d(right_shoulder, right_elbow, right_foot),
                        calculate_angle_3d(right_hip, right_knee, right_elbow),
                        calculate_angle_3d(right_ankle, right_foot, right_knee)
                    ]

                    # 피드백 제공
                    feedback = provide_feedback(angles_left, angles_right, target_angles)
                    feedback_collected = True

                # 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 화면 출력
            cv2.imshow(f"{move_name} 자세 피드백", frame)

            # q를 눌러서 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    display_feedback(feedback)

# tkinter GUI 설정
root = tk.Tk()
root.title("자세 선택")
root.geometry("600x600")
root.configure(bg="#f0f0f0")

style = ttk.Style()
style.configure("TButton", padding=15, relief="flat", background="#4CAF50", foreground="black", font=("Helvetica", 12))
style.map("TButton", background=[("active", "#45a049")])

ttk.Label(root, text="동작을 선택하세요:", font=("Helvetica", 18), background="#f0f0f0").pack(pady=20)

button_frame = ttk.Frame(root, padding=10)
button_frame.pack(pady=10)

moves = ['잽', '훅', '어퍼', '스트레이트']
for i, move in enumerate(moves):
    btn = ttk.Button(button_frame, text=move, command=lambda m=move: start_pose_feedback(m))
    btn.grid(row=i//2, column=i%2, padx=10, pady=10, sticky="ew")

# GUI 실행
root.mainloop()
