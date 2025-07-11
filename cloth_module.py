def run_cloth_analysis():
    import cv2
    from ultralytics import YOLO
    import numpy as np
    import time
    from collections import defaultdict

    # Load the model
    model = YOLO(r"C:\Users\USER\Downloads\best.pt")

    # ✅ Warm-up model to avoid lag
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = model.predict(dummy_frame, imgsz=480, verbose=False)

    # Constants
    CONFIDENCE_THRESHOLD = 0.6
    CLASS_NAMES = model.names
    IGNORE_CLASSES = []

    # Set up camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Colors for bounding boxes
    np.random.seed(42)
    BOX_COLORS = {
        cls_id: tuple(np.random.randint(0, 150, 3).tolist())
        for cls_id in range(len(CLASS_NAMES))
    }

    # Utility functions
    def get_text_color(box_color):
        brightness = 0.299 * box_color[2] + 0.587 * box_color[1] + 0.114 * box_color[0]
        return (255, 255, 255) if brightness < 100 else (0, 0, 0)

    def normalize_labels(labels):
        normalized = set()
        for label in labels:
            if label == "Trouser":
                normalized.add("Pants")
            else:
                normalized.add(label)
        return normalized

    # Define persona rules
    persona_requirements = {
        "Tel Aviv": {"T-shirt", "casual shoes", "Pants"},
        "Haredi": {"Tzitzit", "Kippah"},
        "Ares": {"Necklace", "casual shoes"},
        "Warrior Person": {"Pants", "T-shirt", "Sandals"},
        "The Classic Persona": {"Pants", "T-shirt"},
        "The Free Mind": {"Sandals", "T-shirt"},
    }

    def detect_persona_confidence(item_counter):
        persona_scores = {key: 0.0 for key in persona_requirements}
        
        for persona, required_items in persona_requirements.items():
            matched_items = required_items & item_counter.keys()
            for item in matched_items:
                persona_scores[persona] += item_counter[item]

        final_persona = max(persona_scores, key=persona_scores.get)
        final_score = round(persona_scores[final_persona], 2)

        print("\n🔍 Persona Confidence Scores:")
        for p, s in persona_scores.items():
            print(f"{p}: {s:.2f}")

        if final_score == 0.0:
            final_persona = "Unknown"

        print(f"\n✅ Final Persona: {final_persona} (Confidence Score: {final_score:.2f})")
        return final_persona, final_score

    # State variables
    analysis_started = False
    analysis_completed = False
    start_time = 0
    item_counter = defaultdict(int)
    display_message = "Press SPACE to start analysis"
    final_persona = ""
    final_score = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        current_frame_items = set()

        # Run YOLO detection
        if analysis_started:
            results = model.predict(frame, imgsz=480, verbose=False)
            result = results[0]

            if result.boxes:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    cls_name = CLASS_NAMES.get(cls_id, "Unknown")

                    if conf >= CONFIDENCE_THRESHOLD and cls_name not in IGNORE_CLASSES:
                        current_frame_items.add(cls_name)

                        # Draw bounding box
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        color = BOX_COLORS[cls_id]
                        text_color = get_text_color(color)

                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                        cv2.putText(
                            frame,
                            f"{cls_name} {conf:.2f}",
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            text_color,
                            2,
                        )

            # Count occurrences
            for item in normalize_labels(current_frame_items):
                item_counter[item] += 1

            # Time check
            elapsed = time.time() - start_time
            if elapsed >= 6:
                final_persona, final_score = detect_persona_confidence(item_counter)
                analysis_started = False
                analysis_completed = True
                display_message = f"Persona: {final_persona} | Score: {final_score:.2f} | Press R to reset"

        # Display message
        if not analysis_started and not analysis_completed:
            display_message = "Press SPACE to start analysis"
        elif analysis_started:
            display_message = "Analyzing..."

        (text_width, _), _ = cv2.getTextSize(display_message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (frame_width - text_width) // 2
        text_y = frame_height // 2

        cv2.putText(
            frame,
            display_message,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if "Unknown" not in display_message else (0, 0, 255),
            2,
        )

        cv2.imshow("Clothing-Based Persona Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and not analysis_started and not analysis_completed:
            print("⏳ Analysis started...")
            item_counter.clear()
            start_time = time.time()
            analysis_started = True
            final_persona = ""
            final_score = 0.0
        elif key == ord('r') and not analysis_started:
            print("🔁 Resetting analysis...")
            analysis_completed = False
            item_counter.clear()
            final_persona = ""
            final_score = 0.0
            display_message = "Press SPACE to start analysis"

    cap.release()
    cv2.destroyAllWindows()
    return {final_persona: final_score}