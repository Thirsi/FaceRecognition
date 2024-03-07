from cv2 import FONT_HERSHEY_COMPLEX, imshow, waitKey, putText, cvtColor, COLOR_BGR2RGB, rectangle
import face_recognition

imgHenry = face_recognition.load_image_file('ImagesBasic/Henry.jpg')
imgHenry = cvtColor(imgHenry, COLOR_BGR2RGB)

#Change the commented line to pass/fail the test
#imgTest = face_recognition.load_image_file('ImagesBasic/Henry_Test2.jpg') #Pass
imgTest = face_recognition.load_image_file('ImagesBasic/Henry_Test.jpg') #Pass
#imgTest = face_recognition.load_image_file('ImagesBasic/Elon_Test.jpg') #Fail

imgTest = cvtColor(imgTest, COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgHenry)[0]
encodeHenry = face_recognition.face_encodings(imgHenry)[0]
rectangle(imgHenry, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeHenry], encodeTest)
faceDis = face_recognition.face_distance([encodeHenry], encodeTest)
print(results, faceDis)
putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

imshow('Henry', imgHenry)
imshow('Henry Test', imgTest)
waitKey(0)
