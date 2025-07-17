import cv2
import numpy as np
import os
import time

# Google Colab specific Code Snippet
# This code is designed to run in Google Colab environment.
# --- 1. Install Ultralytics YOLOv8 ---
!pip install ultralytics
print("Ultralytics installation complete.")
from ultralytics import YOLO # Import YOLO after installation


# --- 2. Configuration Parameters ---
# IMPORTANT: Adjust these parameters for your specifications

# Define the specific error class you want to trigger an alert for
# Ensure this exact string matches one of your YOLO model's class names (e.g., in data.yaml or model.names)
TARGET_ERROR_CLASS = "Winding_error"

# Minimum confidence threshold (from your ML model) to trigger an alert
ALERT_CONFIDENCE_THRESHOLD = 0.7

# Path to your YOLOv8 model weights.
MODEL_WEIGHTS_PATH = '/content/best.pt' # Ensure this is your custom model or a downloaded official one

# Path to your uploaded input video file in Colab's session storage
VIDEO_INPUT_PATH = '/content/overlapping2.mp4' # <--- Make sure your uploaded video is named this!

# Path for the output video file
VIDEO_OUTPUT_PATH = '/content/output_video_with_alerts.mp4'

# --- Alert Display Parameters ---
# ADJUSTED: Smaller font scale for the alert text
ALERT_FONT_SCALE = 0.5
ALERT_FONT_THICKNESS = 2     # Thickness of the alert text
ALERT_RIGHT_MARGIN = 20      # Pixels from the right edge of the frame
ALERT_TOP_MARGIN = 20        # Pixels from the top edge of the frame
ALERT_LINE_SPACING = 5       # Pixels between the two lines of alert text

# --- 3. Model Weights Management ---
if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"Model weights not found at {MODEL_WEIGHTS_PATH}. Attempting to download yolov8n.pt as this path...")
    !wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -O {MODEL_WEIGHTS_PATH}
    print(f"Downloaded yolov8n.pt and saved as {MODEL_WEIGHTS_PATH}.")
else:
    print(f"Model weights found at {MODEL_WEIGHTS_PATH}.")

# --- 4. Load the YOLOv8 Model (ONLY ONCE) ---
print(f"Loading YOLOv8 model from {MODEL_WEIGHTS_PATH}...")
try:
    model = YOLO(MODEL_WEIGHTS_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    print("Please check if the model path is correct and the file exists.")
    exit() # Exit if the model can't be loaded

# --- 5. Main Inspection Pipeline Logic ---
def run_inspection_pipeline():
    """
    This function performs continuous inspection using the loaded YOLOv8 model,
    draws detections and alerts on frames, and saves to an output video.
    """
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open input video file at {VIDEO_INPUT_PATH}.")
        print("Please ensure you have uploaded the video file to the Colab session storage and that VIDEO_INPUT_PATH is correct.")
        return # Exit if video cannot be opened

    # Get video properties for VideoWriter
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Corrected from previous version
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30 # Default to 30 FPS if not detected

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 'mp4v' or 'XVID' often work well for .mp4.
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open output video file for writing at {VIDEO_OUTPUT_PATH}.")
        print("Check if the path is valid and you have write permissions.")
        cap.release()
        return # Exit if output video can't be opened

    frame_num = 0
    alert_triggered_for_target_error = False # Flag for first detection

    print(f"Successfully opened input video: {VIDEO_INPUT_PATH}. FPS: {fps}")
    print(f"Writing output video to: {VIDEO_OUTPUT_PATH} (Resolution: {frame_width}x{frame_height}, FPS: {fps})")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video stream or error reading frame.")
            break # Break loop if video ends or error occurs

        frame_num += 1

        # --- Run Actual YOLOv8 Inference ---
        if frame is None or frame.size == 0:
            print(f"Warning: Received empty frame at frame_num {frame_num}. Skipping inference.")
            continue

        results = model.predict(frame, verbose=False, conf=ALERT_CONFIDENCE_THRESHOLD, iou=0.5, classes=None)

        # --- Process Detections and Draw ---
        fault_detected_in_frame = False
        display_frame = frame.copy() # Make a copy to draw on

        current_detection_confidence = 0.0
        detected_fault_name = ""

        for r in results:
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    class_id = int(r.boxes.cls[i])
                    confidence = float(r.boxes.conf[i])
                    bbox_xywh = r.boxes.xywhn[i].cpu().numpy()

                    detected_class_name = r.names[class_id]

                    h_img, w_img, _ = frame.shape
                    center_x, center_y, box_w, box_h = bbox_xywh
                    x1 = int((center_x - box_w/2) * w_img)
                    y1 = int((center_y - box_h/2) * h_img)
                    x2 = int((center_x + box_w/2) * w_img)
                    y2 = int((center_y + box_h/2) * h_img)

                    # Draw bounding box and label for ALL detections that meet confidence
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green for general detections
                    text_label = f"{detected_class_name} {confidence:.2f}"
                    cv2.putText(display_frame, text_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Check if this detection is our TARGET_ERROR_CLASS
                    if detected_class_name == TARGET_ERROR_CLASS:
                        fault_detected_in_frame = True
                        current_detection_confidence = confidence
                        detected_fault_name = detected_class_name
                        # Draw a red box for the specific target error
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red for target error
                        cv2.putText(display_frame, f"FAULT: {detected_class_name}", (x1, y1 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- Draw On-Video Alert Text in Top-Right Corner ---
        if fault_detected_in_frame:
            # Prepare the two lines of text
            line1_text = f"Fault detected: {detected_fault_name.upper()}"
            line2_text = f"Prob: {current_detection_confidence*100:.2f}%"
            alert_color = (0, 0, 255) # Red for alert

            # Get text sizes for correct positioning
            (line1_width, line1_height), _ = cv2.getTextSize(line1_text, cv2.FONT_HERSHEY_SIMPLEX, ALERT_FONT_SCALE, ALERT_FONT_THICKNESS)
            (line2_width, line2_height), _ = cv2.getTextSize(line2_text, cv2.FONT_HERSHEY_SIMPLEX, ALERT_FONT_SCALE, ALERT_FONT_THICKNESS)

            # Determine the widest line for background box
            max_text_width = max(line1_width, line2_width)
            total_text_height = line1_height + ALERT_LINE_SPACING + line2_height

            # Calculate top-right corner coordinates for the text block
            # x-coordinate for right alignment
            text_block_x1 = frame_width - max_text_width - ALERT_RIGHT_MARGIN
            # y-coordinate for top alignment
            text_block_y1 = ALERT_TOP_MARGIN

            # Calculate coordinates for the background rectangle
            # Adjusted padding (was 10, now 5 for smaller text)
            rect_x1 = text_block_x1 - 5
            rect_y1 = text_block_y1 - line1_height - 5
            rect_x2 = frame_width - ALERT_RIGHT_MARGIN + 5
            rect_y2 = text_block_y1 + total_text_height + 5

            # Draw background rectangle
            cv2.rectangle(display_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1) # Black background

            # Position and draw line 1
            line1_x = frame_width - line1_width - ALERT_RIGHT_MARGIN # Align line1 to the right
            line1_y = text_block_y1
            cv2.putText(display_frame, line1_text, (line1_x, line1_y),
                        cv2.FONT_HERSHEY_SIMPLEX, ALERT_FONT_SCALE, alert_color, ALERT_FONT_THICKNESS, cv2.LINE_AA)

            # Position and draw line 2
            line2_x = frame_width - line2_width - ALERT_RIGHT_MARGIN # Align line2 to the right
            line2_y = line1_y + line1_height + ALERT_LINE_SPACING
            cv2.putText(display_frame, line2_text, (line2_x, line2_y),
                        cv2.FONT_HERSHEY_SIMPLEX, ALERT_FONT_SCALE, alert_color, ALERT_FONT_THICKNESS, cv2.LINE_AA)

            # First detection logic (for console output)
            if not alert_triggered_for_target_error:
                print(f"!!! FIRST INSTANCE OF {detected_fault_name.upper()} DETECTED at Frame: {frame_num} (Confidence: {current_detection_confidence:.2f}) !!!")
                alert_triggered_for_target_error = True
        else:
            # If target fault is NOT detected, reset the flag
            if alert_triggered_for_target_error:
                print(f"Frame {frame_num}: {TARGET_ERROR_CLASS} no longer detected. Resetting alert state.")
            alert_triggered_for_target_error = False

        # Write the processed frame to the output video
        out.write(display_frame)

    # Release everything when the job is finished
    cap.release()
    out.release()
    print("\n--- Inspection Pipeline Finished ---")
    print(f"Output video saved to: {VIDEO_OUTPUT_PATH}")
    print("You can download this video from the Colab 'Files' tab.")

# --- Run the Inspection Pipeline ---
if __name__ == '__main__':
    run_inspection_pipeline()