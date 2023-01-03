import face_recognition
import cv2
import numpy as np
import csv#comma seperated files
import os #to acess all the fies in the system
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#loading the images + encoding(reading all the data)
anisha_image =face_recognition.load_image_file("images/anisha.jpg")
anisha_encoding = face_recognition.face_encodings(anisha_image)[0]

chakrika_image =face_recognition.load_image_file("images/chakrika.jpg")
chakrika_encoding = face_recognition.face_encodings(chakrika_image)[0]

known_faces_encoding = [anisha_encoding,chakrika_encoding]

known_faces_name = ["anisha", "chakrika", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]

Room = known_faces_name.copy()
face_location = []#stores the loc of face - coordinates
face_encoding = []#data encoding
face_names =[]#names storing
s=True

now = datetime.now()
current_data = now.strftime("%d - %m - %y")

#writing the file
f = open(current_data + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()#video input reading
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)#decreasing the size
    rgb_small_frame = small_frame[:,:,::-1]#bgr-->rgb
    if s :
        face_locations = face_recognition.face_locations(rgb_small_frame)#recognises presence of a face in the frame
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)#store the data of the coming frame
        face_names=[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces_encoding,face_encoding)
            name = "Unknown"
            face_distance = face_recognition.face_distance(known_faces_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)#?////
            if matches[best_match_index]:
                name = known_faces_name[best_match_index]

            face_names.append(name)
            if name in known_faces_name:
                if name in Room:
                    Room.remove(name)
                    print(Room)
                    current_time = now.strftime("%H - %M - %S")
                    #current_date = now.strftime("")
                    lnwriter.writerow([name, current_data])
    cv2.imshow("register",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()