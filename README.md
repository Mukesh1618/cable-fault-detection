![Python](https://img.shields.io/badge/Python-3.8+-blue)

![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-green)

![Status](https://img.shields.io/badge/Status-Active-success)

🚀 Cable Fault Detection using YOLOv8

This project implements a real-time wire/cable fault detection system using deep learning. It leverages the YOLOv8 object detection model and OpenCV to automatically detect faults in cable windings from live camera feed or recorded video.

Developed as part of an internship project, the system focuses on accurate and fast detection to assist in automated inspection workflows.

🎯 Key Features

✅ Real-time fault detection

✅ Supports live camera and video input

✅ YOLOv8-based deep learning model

✅ Automatic annotation of detected faults

✅ High accuracy (~90%+)

✅ Lightweight and fast processing

✅ Works on custom client dataset

🧠 How It Works

Video input is captured (live camera or file).

Frames are processed using OpenCV.

YOLOv8 model analyzes each frame.

Faults in cable windings are detected.

Bounding boxes and labels are drawn on the output video.

Annotated video stream is displayed/saved.

🛠️ Tech Stack

Python

OpenCV

YOLOv8 (Ultralytics)

Deep Learning / Computer Vision
