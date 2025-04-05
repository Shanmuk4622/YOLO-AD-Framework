import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cvzone
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the fine-tuned model
model = YOLO('models/v10m.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load image
image_path = "acci_1012.jpg"  # Change to your image path
frame = cv2.imread(image_path)

# Load class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

frame = cv2.resize(frame, (1020, 500))

# Start the timer for computation
computation_start = time.time()

# Perform object detection
prediction_start = time.time()
results = model.predict(frame)
prediction_end = time.time()

# End the timer for computation
computation_end = time.time()

# Calculate time taken
prediction_time = prediction_end - prediction_start
computation_time = computation_end - computation_start

print("\n--- Performance Metrics ---")
print(f"Prediction Time: {prediction_time:.4f} seconds")
print(f"Total Computation Time: {computation_time:.4f} seconds")

# Extract detection metrics
detection_speed = results[0].speed
preprocess_time = detection_speed['preprocess']
inference_time = detection_speed['inference']
postprocess_time = detection_speed['postprocess']

print("\n--- Detection Speed ---")
print(f"Preprocess Time: {preprocess_time:.4f} ms")
print(f"Inference Time: {inference_time:.4f} ms")
print(f"Postprocess Time: {postprocess_time:.4f} ms")

# Extract detected objects
a = results[0].boxes.data
px = pd.DataFrame(a).astype("float")
num_objects = len(px)
print("\n--- Detected Objects ---")
print(f"Total Objects Detected: {num_objects}\n")

true_labels = []  # Ground truth labels (To be provided manually or from dataset)
predicted_labels = []  # Model predictions

for index, row in px.iterrows():
    x1 = int(row[0])
    y1 = int(row[1])
    x2 = int(row[2])
    y2 = int(row[3])
    confidence = float(row[4])  # Confidence score
    d = int(row[5])
    c = class_list[d]

    print(f"Object {index + 1}:")
    print(f"    Class       : {c}")
    print(f"    Confidence  : {confidence:.2f}")
    print(f"    Coordinates : ({x1}, {y1}), ({x2}, {y2})\n")

    # Store predicted labels
    predicted_labels.append(d)

    # Draw rectangle and label
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cvzone.putTextRect(frame, f'{c} ({confidence:.2f})', (x1, y1), 1, 1)

# # Define ground truth labels (Example: Manually setting labels for accuracy calculation)
# true_labels = [class_list.index("accident")] * len(predicted_labels)  # Replace with actual ground truth labels
#
# # Compute accuracy, precision, recall, and F1-score
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
# recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
# f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
#
# print("\n--- Model Performance Metrics ---")
# print(f"Accuracy   : {accuracy:.4f}")
# print(f"Precision  : {precision:.4f}")
# print(f"Recall     : {recall:.4f}")
# print(f"F1-score   : {f1:.4f}")

cv2.imshow("RGB", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
