import face_recognition
import cv2
from fer import FER

fer = FER(mtcnn=True)

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_locations = face_recognition.face_locations(gray_frame)

    emotion = fer.predict_emotion(gray_frame)

    for top, right, bottom, left in face_locations:
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
        print("The detected emotion is: {}".format(emotion))

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

