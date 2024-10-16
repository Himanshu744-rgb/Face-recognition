import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import time
from datetime import datetime

# Load the face detection cascade classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if facedetect.empty():
    print("Failed to load the face detection cascade classifier.")
    exit()

# Load the trained model and labels
try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError:
    print("Failed to load the trained model and labels.")
    exit()

# Create a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Create the Attendance folder if it doesn't exist
attendance_folder = "Attendance"
if not os.path.exists(attendance_folder):
    os.makedirs(attendance_folder)

# Initialize the video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Failed to open the video capture.")
    exit()

# Create a list to store the names of people who have already been marked as present
present_today = []

while True:
    # Read a frame from the video capture
    ret, frame = video.read()
    if not ret:
        print("Failed to read a frame from the video capture.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Crop the face from the original frame
        crop_img = frame[y:y+h, x:x+w, :]

        # Resize the cropped face to 50x50 pixels
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict the label of the face
        output = knn.predict(resized_img)

        # Get the current date and time
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        # Check if the attendance file exists
        exist = os.path.isfile("Attendance/Attendance_" + date + ".txt")

        # Create the attendance file if it doesn't exist
        if not exist:
            with open("Attendance/Attendance_" + date + ".txt", "w") as f:
                f.write("")

        # Check if the person has already been marked as present today
        if str(output[0]) not in present_today:
            # Write the attendance to the file
            with open("Attendance/Attendance_" + date + ".txt", "a") as f:
                f.write(str(output[0]) + " - " + str(timestamp) + "\n")

            # Add the person to the list of people who have already been marked as present
            present_today.append(str(output[0]))

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

        # Display the predicted label above the face
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Check for the 'q' key to exit
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()