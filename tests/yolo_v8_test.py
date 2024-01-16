import os

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('/home/xmo/socialnav_xmo/feature_extractor/models/yolov8m-pose.pt')
# model = YOLO('models/yolov8m-pose.pt')  # Load an official Pose model

# Open the video file
video_path = "/home/xmo/socialnav_xmo/yolov8-venv/data/pedestrian_1.mp4"

cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        # results = model.predict(frame)

        # Visualize the results on the frame
        res = results[0]
        annotated_frame = results[0].plot()

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if res.keypoints.shape[0] >= 2:
            print("multi res")

    else:
        # Break the loop if the end of the video is reached
        print('load faid')
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()