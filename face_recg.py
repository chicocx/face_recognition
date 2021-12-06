import cv2
import numpy as np
import face_recognition
import os

images = 'images'

imgElon = face_recognition.load_image_file(f'{images}/elon1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file(f'{images}/elon2.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

imgFrancisco1 = face_recognition.load_image_file(f'{images}/francisco_with_beard.jpg')
imgFrancisco1 = cv2.cvtColor(imgFrancisco1, cv2.COLOR_BGR2RGB)

imgFrancisco2 = face_recognition.load_image_file(f'{images}/francisco_without_beard.jpg')
imgFrancisco2 = cv2.cvtColor(imgFrancisco2, cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 0, 255), 2)

faceTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceTest[3], faceTest[0]), (faceTest[1], faceTest[2]), (0, 0, 255), 2)

faceFrancisco1 = face_recognition.face_locations(imgFrancisco1)[0]
encodeFrancisco1 = face_recognition.face_encodings(imgFrancisco1)[0]
cv2.rectangle(imgFrancisco1, (faceFrancisco1[3], faceFrancisco1[0]), (faceFrancisco1[1], faceFrancisco1[2]), (0, 0, 255), 2)

faceFrancisco2 = face_recognition.face_locations(imgFrancisco2)[0]
encodeFrancisco2 = face_recognition.face_encodings(imgFrancisco2)[0]
cv2.rectangle(imgFrancisco2, (faceFrancisco2[3], faceFrancisco2[0]), (faceFrancisco2[1], faceFrancisco2[2]), (0, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
distance = face_recognition.face_distance([encodeElon], encodeTest)

resultsFranciscoFrancisco = face_recognition.compare_faces([encodeFrancisco1], encodeFrancisco2)
distanceFranciscoFrancisco = face_recognition.face_distance([encodeFrancisco1], encodeFrancisco2)

resultsFranciscoElon = face_recognition.compare_faces([encodeFrancisco1], encodeElon)
distanceFranciscoElon = face_recognition.face_distance([encodeFrancisco1], encodeElon)

print('EncodeElon', encodeElon)
print('EncodeFrancisco', encodeFrancisco1)

print('Elon com Elon', results)
print('Elon com Elon', distance)

print('Francisco com Francisco', resultsFranciscoFrancisco)
print('Francisco com Francisco', distanceFranciscoFrancisco)

print('Francisco com Elon', resultsFranciscoElon)
print('Francisco com Elon', distanceFranciscoElon)

cv2.imshow('elon', imgElon)
cv2.imshow('test', imgTest)

cv2.imshow('Francisco with beard', imgFrancisco1)
cv2.imshow('Francisco without beard', imgFrancisco2)


cv2.waitKey(0)