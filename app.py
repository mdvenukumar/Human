import cv2  # Import the OpenCV library for computer vision
import numpy as np  # Import the NumPy library for numerical operations
import random  # Import the random module for generating random numbers
import os  # Import the os module for interacting with the operating system
from threading import Thread  # Import the Thread class for multithreading

# Create a class named 'Detector' to encapsulate the object detection functionality
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        # Initialize the Detector object with paths to video, configuration, model, and classes files
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Create an object of the cv2.dnn_DetectionModel class with the specified model and configuration
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        # Set input size, scale, mean, and swap RB channels for the neural network
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Create a dictionary to store random colors for each class label
        self.label_colors = {class_label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                             for class_label in self.readClasses()}

    # Method to read classes from the provided classes file
    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            classesList = f.read().splitlines()
        return ['__background__'] + classesList

    # Method to process a frame, detect persons, and return their bounding boxes
    def processFrame(self, frame, margin_factor=0.1, conf_threshold=0.42):
        # Detect objects in the frame using the neural network with specified confidence threshold
        classesLabelIDs, _, bboxs = self.net.detect(frame, confThreshold=conf_threshold)

        # Initialize variables to count persons and store their information
        person_count = 0
        persons = []

        # Check if any bounding boxes are detected
        if bboxs is not None:
            # Iterate through each detected object and its bounding box
            for classLabelID, bbox in zip(classesLabelIDs, bboxs):
                # Convert classLabelID to an integer
                classLabelID = int(classLabelID)
                # Get the corresponding class label using the readClasses method
                classLabel = self.readClasses()[classLabelID]

                # Check if the detected object is a person
                if classLabel.lower() == 'person':
                    # Increment the person count
                    person_count += 1

                    # Extract and adjust the bounding box coordinates with a margin
                    x, y, w, h = bbox
                    margin_x = int(w * margin_factor)
                    margin_y = int(h * margin_factor)
                    x = max(0, x - margin_x)
                    y = max(0, y - margin_y)
                    w = min(frame.shape[1] - x, w + 2 * margin_x)
                    h = min(frame.shape[0] - y, h + 2 * margin_y)

                    # Append the person's information to the list
                    persons.append((x, y, w, h, classLabel))

        # Return the list of persons and the total person count
        return persons, person_count

    # Method to draw bounding boxes and class labels on the frame
    def drawBoundingBoxes(self, frame, persons):
        # Iterate through each person's information in the list
        for person in persons:
            x, y, w, h, classLabel = person
            # Get the color associated with the person's class label
            classColor = self.label_colors.get(classLabel, (255, 255, 255))

            # Draw a rectangle around the person and put the class label above the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=classColor, thickness=1)
            cv2.putText(frame, classLabel, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        # Return the modified frame
        return frame

    # Method to process a video, detect persons, and save the output to a new video file
    def processVideo(self, margin_factor=0.1, conf_threshold=0.42):
        # Open the video file for reading
        cap = cv2.VideoCapture(self.videoPath)
        # Check if the video file is opened successfully
        if not cap.isOpened():
            print("Error opening the video file")
            return

        # Get the frames per second (fps) and frame size of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Create a VideoWriter object to save the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_video.mp4', fourcc, fps, frame_size)

        # Loop through each frame in the video
        while True:
            # Read a frame from the video
            success, frame = cap.read()
            # Check if the frame is read successfully
            if not success:
                break

            # Process the frame to detect persons and get their information
            persons, person_count = self.processFrame(frame, margin_factor, conf_threshold)

            # Display the number of detected persons on the frame
            info_text = f"Detected Persons: {person_count}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw bounding boxes and labels on the frame
            frame_with_boxes = self.drawBoundingBoxes(frame, persons)

            # Write the frame with bounding boxes to the output video file
            out.write(frame_with_boxes)

            # Display the frame with bounding boxes
            cv2.imshow("Object Detection", frame_with_boxes)

            # Check for the 's' key to stop the video processing
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break

        # Release the video capture and video writer objects
        cap.release()
        out.release()
        # Close all OpenCV windows
        cv2.destroyAllWindows()

# Entry point of the script
if __name__ == '__main__':
    try:
        # Prompt the user to choose the mode (camera or video)
        mode = input("Choose mode (1 for camera, 2 for video): ")

        # Set the videoPath based on the user's choice
        if mode == '1':
            videoPath = 0  # Use the default camera
        elif mode == '2':
            videoPath = 'model_data/complete.mp4'  # Use the specified video file
        else:
            print("Invalid mode choice. Please enter '1' for camera or '2' for video.")
            exit()

        # Set the paths for configuration, model, and classes files
        configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
        classesPath = os.path.join("model_data", "coco.names")

        # Create a Detector object with the specified paths
        app = Detector(videoPath, configPath, modelPath, classesPath)

        # Create a thread for video processing
        video_thread = Thread(target=app.processVideo)
        video_thread.start()

        # Wait for the video processing thread to finish
        video_thread.join()

    except Exception as e:
        print(f"An error occurred: {str(e)}")
