import cv2
import numpy as np
import pickle


cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels={"person_name":1}

with open("labels.pickle", 'rb') as f:#to save label ids in a file
	og_labels=pickle.load(f)
	labels={v:k for k,v in og_labels.items()}

video_captutre = cv2.VideoCapture(0) #object for capturing video

while True:
	ret, frame = video_captutre.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#cascade works for gray
	faces = face_cascade.detectMultiScale(gray ,scaleFactor=1.5 , minNeighbors=5) 

	
	for (x,y,w,h) in faces:
		print(x,y,w,h)

	

	roi_gray=gray[y:y+h, x:x+w]
	roi_color=frame[y:y+h ,x:x+w]

	id_ ,conf = recognizer.predict(roi_gray)
	if conf>=45 and conf<=85:
		#print(id_)
		#print(labels[id_])
		font=cv2.FONT_HERSHEY_SIMPLEX
		name=labels[id_]
		color=(255,255,255)
		stroke=2
		cv2.putText(frame,name,(x,y),font , 1 ,color,stroke)#,cv2.LINE_AA)
	image_item='my_image.png'

	cv2.imwrite(image_item,roi_color)#this thing will only save my face and nothig else but in gray
	#dray a rectangle
	color= (255,0,0)

	stroke=2

	cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)
	eyes = eye_cascade.detectMultiScale(gray)
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),stroke)


	cv2.imshow("live",frame)

	if cv2.waitKey(1) and 0xFF == ord('q'):
		break
video_captutre.release()
cv2.DestroyALlWindows()

