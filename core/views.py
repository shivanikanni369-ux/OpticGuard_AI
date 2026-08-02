import cv2
import mediapipe as mp
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render

# Initialize both Pose and Face Mesh tools
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
bad_frames_count = 0

# Track blinks and eye strain variables
blink_counter = 0
eye_strain_frames = 0
is_eye_strain = False
eye_closed_flag = False

# Landmark index pointers for Eye Aspect Ratio (EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

current_status = {
    "posture": "Detecting...",
    "is_bad": False,
    "is_missing": True,
    "is_eye_strain": False,
    "blink_count": 0
}

def index(request):
    return render(request, 'core/index.html')

def get_ear(landmarks, eye_indices, img_w, img_h):
    pts = [np.array([landmarks[i].x * img_w, landmarks[i].y * img_h]) for i in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h != 0 else 0

def gen_frames():
    global bad_frames_count, current_status, blink_counter, eye_strain_frames, is_eye_strain, eye_closed_flag

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        face_results = face_mesh.process(rgb)

        posture = "Detecting..."
        color = (255, 255, 255) 
        is_bad = False
        is_missing = True
        is_eye_strain = False

        if results.pose_landmarks:
            is_missing = False
            lm = results.pose_landmarks.landmark

            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
            r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]
            nose = lm[mp_pose.PoseLandmark.NOSE]

            ls_x, ls_y = int(ls.x * w), int(ls.y * h)
            rs_x, rs_y = int(rs.x * w), int(rs.y * h)
            le_x, le_y = int(l_ear.x * w), int(l_ear.y * h)
            re_x, re_y = int(r_ear.x * w), int(r_ear.y * h)
            n_x, n_y = int(nose.x * w), int(nose.y * h)

            # Proportional distance evaluations
            shoulder_width = np.hypot(ls_x - rs_x, ls_y - rs_y)
            if shoulder_width == 0: 
                shoulder_width = 1

            l_neck_height = ls_y - le_y
            r_neck_height = rs_y - re_y
            neck_height = max(l_neck_height, r_neck_height)

            posture_ratio = neck_height / shoulder_width
            mid_x = (ls_x + rs_x) // 2
            head_offset = abs(n_x - mid_x)
            shoulder_diff = abs(ls_y - rs_y)

            if shoulder_diff > 30 or head_offset > 45 or posture_ratio < 0.38:
                is_bad = True

            if is_bad:
                bad_frames_count += 1
            else:
                bad_frames_count = 0

            if bad_frames_count > 8:
                posture = "Bad Posture"
                color = (0, 0, 255) 
            else:
                posture = "Good Posture"
                color = (0, 255, 0) 
        else:
            posture = "Missing"
            color = (255, 255, 0) 
            bad_frames_count = 0

        # Eye tracking calculations
        if face_results.multi_face_landmarks and not is_missing:
            face_lms = face_results.multi_face_landmarks[0].landmark
            
            left_ear_score = get_ear(face_lms, LEFT_EYE, w, h)
            right_ear_score = get_ear(face_lms, RIGHT_EYE, w, h)
            avg_ear = (left_ear_score + right_ear_score) / 2.0

            if avg_ear < 0.21:
                eye_closed_flag = True
            else:
                if eye_closed_flag:
                    blink_counter += 1
                    eye_closed_flag = False
                    eye_strain_frames = 0

            p1 = np.array([face_lms[133].x * w, face_lms[133].y * h])
            p2 = np.array([face_lms[362].x * w, face_lms[362].y * h])
            eye_distance = np.linalg.norm(p1 - p2)

            if eye_distance > 85: 
                eye_strain_frames += 1

            if eye_strain_frames > 45:
                is_eye_strain = True

        # Synchronize parameters into state dictionary
        current_status = {
            "posture": posture,
            "is_bad": (posture == "Bad Posture"),
            "is_missing": (posture == "Missing"),
            "is_eye_strain": is_eye_strain,
            "blink_count": blink_counter
        }

        cv2.putText(frame, posture, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        if is_eye_strain:
            cv2.putText(frame, "EYE STRAIN ALERT", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')

def get_posture_data(request):
    return JsonResponse(current_status)