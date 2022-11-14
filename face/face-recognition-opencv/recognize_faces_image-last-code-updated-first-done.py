# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png  --output output/example_01_output.png
# python recognize_faces_image.py

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import cv2
import pymsgbox
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import *

#the form and its name 
root = tk.Tk() 
root.title('hi')

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")

args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())
writer = None

#the function to open the image and save it
def open():

	#open the window to select the image
	filename =  filedialog.askopenfilename(initialdir = "/home/abanob/Downloads",title = "Select file",filetypes = (("jpeg files","*.jpg"),("allfiles","*.*")))
	
	#variable used as boolean function 
	u=0

	print (filename)
	
	# load the input image and convert it from BGR to RGB
	image = cv2.imread(filename)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	#rgb = imutils.resize(image, width=750)
	#r = image.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
	boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
	
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		name = "Unknown"
		
		# check to see if we have found a match
		if True in matches:
		
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1
			
			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	# loop over the recognized faces
	for ((top, right, bottom, left), name) in zip(boxes, names):
		
		"""
		#rescale the face coordinates
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)
		"""
		#warning 
		if name == 'abanob_ashraf':

			cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 255), 2)
			u=1  
			
		else:

			# draw the predicted face name on the image
			cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
			y = top - 15 if top - 15 > 15 else top + 15
			cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
	
	# resize image
	# get the dimensions of image before resize --> height, width, number of channels in image
	dimensions = image.shape
	print('Image Dimension    : ',dimensions)
	
	scale_percent = 50 # percent of original size
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)

	resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	
	# get the dimensions of image after resize --> height, width, number of channels in image
	dimensions = resize.shape
	print('Image Dimension    : ',dimensions)

	# show the output image
	cv2.imshow("Image", image) 
	if u==1 :
		pymsgbox.alert('abanob_ashraf has been detected', 'Alert!')

	#waitingkey
	cv2.waitKey(0)

	# Download the image after recognized
	# Open the window to select the image where it will save
	filename =  filedialog.asksaveasfilename(initialdir = "/home/abanob/Downloads",title = "Select file",filetypes = (("jpeg files","*.jpg"),("allfiles","*.*")))
	print (filename)
	cv2.imwrite(filename, image)

#the button to open the function to open the image and save it
button = tk.Button(root, text='open', width=25, command=open).pack()

#the button to close the program
button = tk.Button(root, text='Stop', width=25, command=root.destroy).pack()

#to loop the program
root.mainloop()
