import cv2
import numpy as np
import winsound

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Replace 0 with your video feed
previous_positions = {}

# Constants
SPEED_THRESHOLD = 15  # Threshold for fast-approaching vehicles
STOP_THRESHOLD = 5    # Threshold for detecting sudden stops

def warn_driver(message):
    print(f"WARNING: {message}")
    winsound.Beep(440, 1000)  # Frequency 440 Hz, Duration 1000 ms

def calculate_speed(new_positions, previous_positions):
    speeds = {}
    for obj_id, (x, y, w, h) in new_positions.items():
        if obj_id in previous_positions:
            dx = x - previous_positions[obj_id][0]
            dy = y - previous_positions[obj_id][1]
            speed = np.sqrt(dx**2 + dy**2)  # Approximate relative speed
            speeds[obj_id] = speed
    return speeds

while True:
    _, frame = cap.read()
    height, width, _ = frame.shape

    # Preprocessing for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    new_positions = {}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ['car', 'truck', 'bus', 'motorbike']:
                center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                new_positions[class_id] = (x, y, w, h)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{classes[class_id]}: {int(confidence * 100)}%", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Calculate relative speeds
    speeds = calculate_speed(new_positions, previous_positions)
    for obj_id, speed in speeds.items():
        if speed > SPEED_THRESHOLD:
            warn_driver("Fast-approaching vehicle detected!")
        elif speed < STOP_THRESHOLD:
            warn_driver("Sudden stop detected in front!")

    previous_positions = new_positions

    # Show the frame
    cv2.imshow("Driver Warning System", frame)

    # Exit on pressing ESC
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

