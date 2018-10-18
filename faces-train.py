#os.walk for images
import cv2
import os
from PIL import Image
import numpy as np
import pickle #to save these labels

cascPath = "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascPath)

recognizer = cv2.face.LBPHFaceRecognizer_create() #this will create a in built opencv face recognizer

Base_dir = os.path.dirname(os.path.abspath(__file__)) #this to know the path where this file is
image_dir = os.path.join(Base_dir , "images") #this would add /images into the base_mame
x_train=[]
y_labels = []
current_id = 0
label_ids = {}

for root,dirs,files in os.walk(image_dir): #for travelling into the path
	for file in files:     
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file) #adds /filename in the path
			label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower() #reads the name of the label or the name of the folder 
			#print(label,path)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id = current_id + 1
			id_ = label_ids[label]
			#print(label_ids)

			#x_train.append(path) #adding images into the train 
			#y_labels.append(label) #some number

			pil_image = Image.open(path).convert("L") #save image somewhere and then convert that image into grayscale
			size = (550,550)
			final_image = pil_image.resize(size , Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8") #convert image into array
			#print(image_array)

			faces = face_cascade.detectMultiScale(image_array ,scaleFactor=1.5 , minNeighbors=5) #detect faces
			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w] #region of interst
				x_train.append(roi)
				y_labels.append(id_)



#print(y_labels)
#print(x_train)
with open("labels.pickle", 'wb') as f:#to save label ids in a file
	pickle.dump(label_ids,f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml") #YAML is a human-readable data serialization language. It is commonly used for configuration files, but could be used in many applications where data is being stored or transmitted. 










			






