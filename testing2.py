import streamlit as st
import cv2
import numpy as np

# variables
Known_distance = 30  # Inches
Known_width = 5.7  # Inches

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
ORANGE = (0, 69, 255)


def face_data(image, CallOut, Distance_level):
    """

    This function Detect face and Draw Rectangle and display the distance over Screen

    :param1 Image(Mat): simply the frame
    :param2 Call_Out(bool): If want show Distance and Rectangle on the Screen or not
    :param3 Distance_Level(int): which change the line according the Distance changes(Intractivate)
    :return1  face_width(int): it is width of face in the frame which allow us to calculate the distance and find focal length
    :return2 face(list): length of face and (face paramters)
    :return3 face_center_x: face centroid_x coordinate(x)
    :return4 face_center_y: face centroid_y coordinate(y)

    """

    face_width = 0
    face_x, face_y = 0, 0
    face_center_x = 0
    face_center_y = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        line_thickness = 2
        # print(len(faces))
        LLV = int(h * 0.12)
        # print(LLV)

        # cv2.rectangle(image, (x, y), (x+w, y+h), BLACK, 1)
        cv2.line(image, (x, y + LLV), (x + w, y + LLV), (GREEN), line_thickness)
        cv2.line(image, (x, y + h), (x + w, y + h), (GREEN), line_thickness)
        cv2.line(image, (x, y + LLV), (x, y + LLV + LLV), (GREEN), line_thickness)
        cv2.line(
            image, (x + w, y + LLV), (x + w, y + LLV + LLV), (GREEN), line_thickness
        )
        cv2.line(image, (x, y + h), (x, y + h - LLV), (GREEN), line_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - LLV), (GREEN), line_thickness)

        face_width = w
        face_center = []
        # Drwaing circle at the center of the face
        face_center_x = int(w / 2) + x
        face_center_y = int(h / 2) + y
        if Distance_level < 10:
            Distance_level = 10

        # cv2.circle(image, (face_center_x, face_center_y),5, (255,0,255), 3 )
        if CallOut == True:
            # cv2.line(image, (x,y), (face_center_x,face_center_y ), (155,155,155),1)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (ORANGE), 28)
            cv2.line(image, (x, y - 11), (x + 180, y - 11), (YELLOW), 20)
            cv2.line(image, (x, y - 11), (x + Distance_level, y - 11), (GREEN), 18)

            # cv2.circle(image, (face_center_x, face_center_y),2, (255,0,255), 1 )
            # cv2.circle(image, (x, y),2, (255,0,255), 1 )

        # face_x = x
        # face_y = y

    return face_width, faces, face_center_x, face_center_y

fonts = cv2.FONT_HERSHEY_COMPLEX

# Initialize camera and face detector
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load reference image and calculate focal length
ref_image = cv2.imread("Ref_image.png")
ref_image_face_width, _, _, _ = face_data(ref_image, False, 0)
Focal_length_found = FocalLength(Known_distance, Known_width, ref_image_face_width)

# Streamlit app
st.title("Face Distance Estimation App")

# Function to perform face detection and distance estimation
def process_frame(frame):
    face_width_in_frame, _, _, _ = face_data(frame, False, 0)

    for (face_x, face_y, face_w, face_h) in _:
        if face_width_in_frame != 0:
            Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
            Distance = round(Distance, 2)
            cv2.putText(
                frame,
                f"Distance: {Distance} Inches",
                (face_x - 6, face_y - 6),
                fonts,
                0.5,
                (BLACK),
                2,
            )

    return frame

# Streamlit web app loop
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame for better display in Streamlit
    resized_frame = cv2.resize(frame, (640, 480))

    # Perform face detection and distance estimation
    processed_frame = process_frame(resized_frame)

    # Display the processed frame in Streamlit
    st.image(processed_frame, channels="BGR", use_column_width=True)

    if st.button("Exit"):
        break

# Release the camera
cap.release()
st.balloons()
