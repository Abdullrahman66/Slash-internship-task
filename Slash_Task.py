import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load YOLO model
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

# Load class names
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]

def detect_objects(image):
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_classes = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            detected_classes.append(label)
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, detected_classes

# Streamlit UI
st.title("YOLO Object Detection")
st.write("Upload an image and click the 'Analyze' button to detect objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button("Analyze"):
        st.write("Analyzing...")

        # Convert the uploaded image to a NumPy array and then to BGR format for OpenCV
        image_np = np.array(image.convert('RGB'))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        detected_image, detected_classes = detect_objects(image_np)

        # Convert BGR image back to RGB
        detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

        st.image(detected_image, caption='Detected Image with Bounding Boxes', use_column_width=True)
        st.write("Detected classes:", detected_classes)