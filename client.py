import io
import streamlit as st
import cv2
import requests
import base64
from PIL import Image
import numpy as np
import json
import os
from datetime import datetime
import pandas as pd

save_dir = os.path.join(os.getcwd(), "misclassified")

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

base_url = "0.0.0.0" if os.getenv("container") else "127.0.0.1"
img_endpoint = "/predict/to-img"
b64_img_endpoint = "/predict/to-b64"
json_endpoint = "/predict/to-json"
url_with_img_endpoint = base_url + img_endpoint
url_with_b64_img_endpoint = base_url + b64_img_endpoint
url_with_json_endpoint = base_url + json_endpoint

def get_response(url, image):
    file = {"image": image}
    response = requests.post(url, files=file)
    print(response.status_code)
    return response

# Draw bounding box on image
def visualise_bbox(image, bbox, class_name, confidence, colour=(255, 0, 0)):
    
    img = np.asarray(image)
    img_w, img_h = img.shape[1], img.shape[0]
    smaller_dim = min(img_w, img_h)
    x_min, y_min, x_max, y_max = bbox
    box_thickness = int(smaller_dim/100)
    cv2.rectangle(img, (x_min,y_min), (x_max, y_max), colour, box_thickness)
    (text_width, text_height), _ = cv2.getTextSize(
        f"{class_name}{confidence: .2f}", 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1,
        1,
    )
    cv2.rectangle(
        img, 
        (max(0, x_min), max(0, y_min - int(1.5 * text_height))), 
        (x_min + int(1.2 * text_width), max(int(1.5 * text_height), y_min)),
        colour,
        -1,
    )
    cv2.putText(
        img,
        text=f"{class_name}{confidence: .2f}",
        org=(max(0, x_min + int(0.05 * text_width)), max(int(1 * text_height), y_min - int(0.4 * text_height))),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
        thickness=2
    )
    return img

# Visualise an image with bounding boxes on it
def visualise(image, bboxes):
    colours = [
        (255, 0, 0),
        (255, 165, 0),
        (0, 255, 0),
        (255, 105, 180),
        (64, 224, 208),
        (134, 1, 175),
    ]
    #names = ["1", "2", "3", "4", "5", "6"]
    names = ["person", "potted plant", "book", "tie", "chair", "laptop"]
    palette = {name: colour for name, colour in zip(names, colours)}
    img = image.copy()
    for bbox in bboxes:
        box = (
            int(bbox["xmin"]), 
            int(bbox["ymin"]), 
            int(bbox["xmax"]), 
            int(bbox["ymax"]),
        )
        img = visualise_bbox(
            img, 
            box, 
            bbox["name"], 
            bbox["confidence"], 
            colour=palette.get(bbox["name"], (128, 128, 128))
        )

    return img

# Construct dynamic table
def build_table(bbox_json, classes=[]):

    prediction = pd.DataFrame(bbox_json)
    if prediction.empty:
        counts = pd.DataFrame({"Nothing": 0}, columns=["Counts"])
        
    else:
        counts = prediction["name"].value_counts().to_frame(name="Counts")
        
    if classes:
        counts = pd.DataFrame(
            {"Counts": [counts.to_dict().get(a_class, 0) for a_class in classes]},
            index=classes,
        )
    
    # Hack to display index name
    counts.index.name = "Classes"
    counts.columns.name = counts.index.name
    counts.index.name = None

    return counts

# Text at the top
st.markdown("<h1>Dice Detector App</h1><br>", unsafe_allow_html=True)
st.markdown("<h3>Instructions:</h3>", unsafe_allow_html=True)
st.markdown("""<ol><li>Check the 'Run' box. It can take some seconds to initialise.</li>
    <li>Select 'Test' on the slider.</li>
    <li>If video and table are displayed properly, switch to 'Dice' on the slider.</li>
    <li>When a scene is incorrectly detected, click the 'Report misclassified image' button.</li>
    <li>When finished, uncheck the 'Run' box.""",
     unsafe_allow_html=True)

# Create layout for display
input_col1, input_col2, input_col3 = st.columns([1, 2, 3])
with input_col1:
    run = st.checkbox("Run")
with input_col2:
    detect_mode = st.select_slider(
        "Run Test before Dice",
        options=["Test", "Dice"],
    )
with input_col3:
    save = st.button("Report misclassified image")

output_col1, output_col2 = st.columns([3, 1])
with output_col1:
    with Image.open("loading-screen.jpg") as im:
        stream = st.image(im)
with output_col2:
    table = st.table()

status = st.empty()
state = "Loading..."
status.write(f"Status: {state}")

# Initialise camera
if "cam" not in st.session_state:
    st.session_state["cam"] = cv2.VideoCapture(0)

if not st.session_state["cam"].isOpened():
    state = "No camera detected"
    raise IOError("Cannot open")
else:
    state = "Initialised"
status.write(f"Status: {state}")

# Run detection
while run:
    _, frame = st.session_state["cam"].read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    _, frame_encoded = cv2.imencode(".jpg", frame)
    frame_bytes = frame_encoded.tobytes()
    frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
    stream_input = json.dumps({
        "b64str": frame_b64,
        "model": detect_mode,
    })
    
    # Display returned image
    # prediction = requests.post(url_with_img_endpoint, data=stream_input)
    # image_stream = io.BytesIO(prediction.content)
    # image_stream.seek(0)
    # file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # stream.image(img)

    # Draw bounding boxes on original image
    prediction = requests.post(url_with_json_endpoint, data=stream_input)
    boxes = prediction.json()
    img = visualise(frame, boxes)
    stream.image(img)
    classes = ["1", "2", "3", "4", "5", "6"] if detect_mode == "Dice" else []
    table.write(build_table(boxes, classes).to_html(), unsafe_allow_html=True)

    if save:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        filename_original = timestamp + "-o.jpg"
        filename_labelled = timestamp + "-l.jpg"
        image_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_labelled = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_dir, filename_original), image_original)
        cv2.imwrite(os.path.join(save_dir, filename_labelled), image_labelled)
        st.write(f"{filename_original} and {filename_labelled} saved")
        save = False

    state = "Running"
    status.write(f"Status: {state}")
else:
    st.session_state["cam"].release()
    del st.session_state["cam"]
    state = "Idle"
    status.write(f"Status: {state}")
