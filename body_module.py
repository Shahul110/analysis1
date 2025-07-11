def run_body_analysis():
    import cv2
    import mediapipe as mp
    import time
    import numpy as np
    import math
    from collections import deque


    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    # Open webcam
    cap = cv2.VideoCapture(0)

    # Previous frame data
    prev_x_pixel = None
    prev_time = None
    speed_history = []

    # Constants
    PIXEL_TO_METER_RATIO = 1 / 300
    SLOW_SPEED_THRESHOLD = 0.9
    FAST_SPEED_THRESHOLD = 1.8
    FEMININE_SWAY_RATIO_THRESHOLD = 0.6
    SPINE_UPRIGHT_ANGLE_THRESHOLD = 5  # degrees from vertical

    # Required landmarks for body visibility
    required_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]

    def calculate_spine_angle(x1, y1, x2, y2):
        # Angle between shoulder-hip line and vertical (y-axis)
        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dx, dy)  # dx, dy to get tilt relative to vertical
        angle_deg = abs(math.degrees(angle_rad))
        return angle_deg
    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20  # or set according to your webcam (usually 20-30)
    def calculate_angle(a, b, c):
        # a, b, c are (x, y) tuples: a = hip, b = knee, c = ankle
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle

    #hands
    def calc_distance(p1, p2):
        return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Define MP4 codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (1280, 720))



    prev_left_wrist = None
    prev_right_wrist = None
    hand_movement_count = 0
    frame_window = deque(maxlen=30)

    analysis_started = False
    analysis_start_time = None
    analysis_duration = 6  # seconds
    final_persona = None

    analysis_time_elapsed = 0
    is_body_visible = False
    paused = False
    waiting_for_visibility = False  # Wait for full body before starting analysis
    first_analysis_started = False  # Add this at the top near other flags

    # Persona score tracking (global for consistent updates)
    persona_scores = {
        "Warrior Person": 0.0,
        "The Classic Persona": 0.0,
        "The Free Mind": 0.0,
        "Tel Aviv": 0.0,
        "Arses": 0.0
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)
        current_time = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and not analysis_started and not waiting_for_visibility and final_persona is None:  # Press spacebar to start analysis
            
            waiting_for_visibility = True
            analysis_started = True
            analysis_start_time = time.time()
            analysis_time_elapsed = 0
            
            print("🔍 Analysis Started")
        elif key == ord('r'):  # Reset
            analysis_started = False
            waiting_for_visibility = False
            final_persona = None
            analysis_time_elapsed = 0
            first_analysis_started = True  # Add this at the top near other flags
            for k in persona_scores:
                persona_scores[k] = 0.0
            print("🔄 Analysis Reset")
        elif key == ord('q'):  # Quit
            break
            
        

        # If analysis not started, show idle screen
        if not analysis_started and final_persona is None and not waiting_for_visibility:
            # After reset
            cv2.putText(frame, "Press SPACE to start analysis", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Persona & Walk Analysis", frame)
            out.write(frame)
            continue
        if not analysis_started and final_persona is not None:
            cv2.putText(frame, f"Final Persona: {final_persona} | Press 'R' to restart", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Persona & Walk Analysis", frame)
            out.write(frame)
            continue
        
        message = ""

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Check full body visibility
            full_body_detected = all(landmarks[lm].visibility > 0.5 for lm in required_landmarks)
            is_body_visible = full_body_detected
            if full_body_detected:
                
                # In each frame
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                left_index = landmarks[mp_pose.PoseLandmark.LEFT_INDEX]
                right_index = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX]

                # Calculate movements
                if prev_left_wrist:
                    if calc_distance(left_wrist, prev_left_wrist) > 0.02:
                        hand_movement_count += 1
                if prev_right_wrist:
                    if calc_distance(right_wrist, prev_right_wrist) > 0.02:
                        hand_movement_count += 1

                frame_window.append(hand_movement_count)

                # Reset counter for next frame
                hand_movement_count = 0

                # Check total movements
                total_movements = sum(frame_window)
                prev_left_wrist = left_wrist
                prev_right_wrist = right_wrist
                
                # Get landmark points
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
                avg_knee_y = (left_knee.y + right_knee.y) / 2
                left_hip1 = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                left_knee1 = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
                left_ankle1 = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)

                right_hip1 = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y)
                right_knee1 = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y)
                right_ankle1 = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)

                # Knee angles
                left_knee_angle = calculate_angle(left_hip1, left_knee1, left_ankle1)
                right_knee_angle = calculate_angle(right_hip1, right_knee1, right_ankle1)
                is_sitting = (60 <= left_knee_angle <= 120) and (60 <= right_knee_angle <= 120)
                
                #stable stand
                # Calculate knee angles
                left_knee_angle3 = calculate_angle(left_hip1, left_knee1, left_ankle1)
                right_knee_angle3 = calculate_angle(right_hip1, right_knee1, right_ankle1)

                # Calculate ankle distance
                ankle_distance = abs(left_ankle.x - right_ankle.x)

                # Standing stable when knees are straight and ankle distance is normal (adjust 0.2 - 0.5 range)
                knees_straight = (160 <= left_knee_angle3 <= 180) and (160 <= right_knee_angle3 <= 180)
                feet_together = ankle_distance < 0.1 and ankle_distance > 0.02 or 0.01 < ankle_distance < 0.02

                # Final check
                is_standing_stable = knees_straight and feet_together
                
                #hands
                hand_results = hands.process(rgb_frame)

                left_hand_open = False
                right_hand_open = False

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                        # Check if hand is open
                        finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                    mp_hands.HandLandmark.RING_FINGER_TIP,
                                    mp_hands.HandLandmark.PINKY_TIP]
                        finger_dips = [mp_hands.HandLandmark.INDEX_FINGER_DIP,
                                    mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                                    mp_hands.HandLandmark.RING_FINGER_DIP,
                                    mp_hands.HandLandmark.PINKY_DIP]

                        extended_fingers = 0
                        for tip, dip in zip(finger_tips, finger_dips):
                            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:  # Finger extended
                                extended_fingers += 1

                        if extended_fingers >= 3:  # Considered open if 3+ fingers extended
                            if handedness.classification[0].label == 'Left':
                                left_hand_open = True
                            else:
                                right_hand_open = True

                
                # ✅ Walking Speed
                current_x = (left_hip.x + right_hip.x) / 2
                current_x_pixel = current_x * frame_width

                if prev_x_pixel is not None and prev_time is not None:
                    delta_pixels = abs(current_x_pixel - prev_x_pixel)
                    delta_time = current_time - prev_time
                    distance_meters = delta_pixels * PIXEL_TO_METER_RATIO
                    speed_mps = distance_meters / delta_time if delta_time > 0 else 0

                    speed_history.append(speed_mps)
                    if len(speed_history) > 5:
                        speed_history.pop(0)
                    avg_speed = np.mean(speed_history)

                    # Classify speed
                    if avg_speed < SLOW_SPEED_THRESHOLD:
                        speed_text = "Too Slow"
                        persona = "Tel Aviv"
                    elif avg_speed > FAST_SPEED_THRESHOLD:
                        speed_text = "Too Fast"
                        persona = "Warrior Person"
                    else:
                        speed_text = "Normal"
                        persona = "The Classic Persona"

                    # ✅ Feminine Walk Detection
                    hip_sway = abs(left_hip.x - right_hip.x)
                    shoulder_sway = abs(left_shoulder.x - right_shoulder.x)
                    sway_ratio = hip_sway / (shoulder_sway + 1e-6)
                    print(f"Hip sway: {hip_sway:.4f}, Shoulder sway: {shoulder_sway:.4f}, Ratio: {sway_ratio:.4f}")
                    # ✅ Spine Upright Detection
                    avg_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
                    avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
                    avg_hip_x = (left_hip.x + right_hip.x) / 2
                    avg_hip_y = (left_hip.y + right_hip.y) / 2

                    spine_angle = calculate_spine_angle(avg_shoulder_x, avg_shoulder_y, avg_hip_x, avg_hip_y)
                    
                    if abs(spine_angle) < SPINE_UPRIGHT_ANGLE_THRESHOLD:
                        spine_status = "Upright"
                        spine_is_upright = True
                    else:
                        spine_status = "Not Upright"
                        spine_is_upright = False

                    # ✅ Final Persona Decision
                    walking_style="unknown"
                    if spine_is_upright:
                        persona ="unknown"
                    if total_movements > 20 and spine_is_upright:
                        persona = "Arses"
                        persona_scores["Arses"] += 0.3
                    if is_sitting and spine_is_upright:
                        persona = "The Classic Persona"
                        persona_scores["The Classic Persona"] += 0.3
                    elif is_standing_stable and spine_is_upright:
                        persona = "Warrior Person"
                        persona_scores["Warrior Person"] += 0.3
                    elif (sway_ratio > FEMININE_SWAY_RATIO_THRESHOLD) and (left_hand_open and right_hand_open):
                        walking_style = "Feminine Walk"
                        #if not spine_is_upright:
                        persona = "The Free Mind"
                        persona_scores["The Free Mind"] += 0.5
                    else:
                        walking_style = "Neutral/Masculine Walk"
                        if speed_text == "Normal" and spine_is_upright:
                            persona = "The Classic Persona"
                            persona_scores["The Classic Persona"] += 0.3
                        elif speed_text == "Too Slow" and not spine_is_upright and total_movements > 5:
                            persona = "Tel Aviv"
                            persona_scores["Tel Aviv"] += 0.4
                        elif speed_text == "Too Fast" and spine_is_upright:
                            persona = "Warrior Person"
                            persona_scores["Warrior Person"] += 0.4
                    if analysis_start_time and (time.time() - analysis_start_time >= analysis_duration):
                        final_persona = persona
                        final_score = persona_scores[final_persona]

                        print("\n🔍 Persona Confidence Scores:")
                        for p, s in persona_scores.items():
                            print(f"{p}: {s:.2f}")
                        print(f"\n✅ Final Persona: {final_persona} (Confidence Score: {final_score:.2f})")

                        analysis_started = False
                        
                        print(f"✅ Final Persona: {final_persona}")
                        cv2.putText(frame, "Press 'r' to start analysis again", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        
                        # Previous wrist positions
                    # ✅ Display Results
                    cv2.putText(frame, f"Speed: {avg_speed:.2f} m/s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Walk: {speed_text}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(frame, f"Persona: {persona}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Style: {walking_style}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(frame, f"Spine: {spine_status}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    print(f"Speed: {avg_speed:.2f} m/s => {speed_text}, Persona: {persona}, Style: {walking_style}, Spine: {spine_status}, Angle: {spine_angle:.2f}°")
                
                # Update for next frame
                prev_x_pixel = current_x_pixel
                prev_time = current_time

            else:
                cv2.putText(frame, "Body Not Fully Visible", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("Body Not Fully Visible")
                prev_x_pixel = None
                prev_time = None
                is_body_visible = False
                speed_history.clear()
            
        
        if waiting_for_visibility:
            if is_body_visible:
                print("✅ Full body visible. Starting 6-second analysis...")
                analysis_started = True
                waiting_for_visibility = False
                analysis_start_time = time.time()
            else:
                message = "Please show full body to begin analysis"
        if analysis_started:
            if is_body_visible:
                if paused:
                    print("✅ Body visible again. Resuming analysis...")
                    paused = False

                # Accumulate time only when visible
                analysis_time_elapsed += 1 / fps

                if analysis_time_elapsed >= analysis_duration:
                    final_persona = persona
                    final_score = persona_scores[final_persona]

                    print("\n🔍 Persona Confidence Scores:")
                    for p, s in persona_scores.items():
                        print(f"{p}: {s:.2f}")
                    print(f"\n✅ Final Persona: {final_persona} (Confidence Score: {final_score:.2f})")

                    print(f"✅ Final Persona: {final_persona}")
                    analysis_started = False
            
                    
                    
            else:
                if not paused:
                    print("⏸️ Body not fully visible. Pausing timer...")
                    paused = True
                message = "Stay visible for analysis"
        
        
        
        else:
            cv2.putText(frame, "No Pose Detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            prev_x_pixel = None
            prev_time = None
            is_body_visible = False
            speed_history.clear()
        resized_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow("Persona & Walk Analysis", resized_frame)

        #cv2.imshow("Persona & Walk Analysis", frame)
    
        out.write(resized_frame)
        

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    return persona_scores

