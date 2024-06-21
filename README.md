# YOLO Object Detection Streamlit App
This Streamlit application allows users to detect objects in an uploaded image using the YOLO (You Only Look Once) object detection algorithm.

## Setup and Installation
**Download YOLO Weights and Config Files**:
    - Download the YOLOv3 weights from the [official website](https://pjreddie.com/media/files/yolov3.weights).
    - Download the YOLOv3 configuration file (yolov3.cfg) from the [official repository](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg).
    - Download the COCO names file (coco.names) from the [official repository](https://github.com/pjreddie/darknet/blob/master/data/coco.names).

    Place these files in the same directory as your Streamlit app.

4. **Run the Streamlit App**:
    ```sh
    streamlit run yolo_streamlit_app.py
    ```
    
## References
- [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
- [Streamlit Documentation](https://docs.streamlit.io/)
