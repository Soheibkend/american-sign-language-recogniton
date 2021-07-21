import cv2, pickle
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from threading import Thread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#model = load_model('model_cnn_v5_seg.h5')
model = load_model('model_cnn_v5_rev_seg.h5')
model_nb = load_model('model_cnn_number.h5')

background = None
accumulated_weight = 0.5

ROI_top = 150
ROI_bottom = 350
ROI_right = 50
ROI_left = 250
#global flag

cam = None
text = " "
word = ""
count_same_frame = 0
num_frames = 0
percentage = 0

def cal_accum_avg(frame, accumulated_weight):

    global background
    
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)



def segment_hand(frame, threshold=10):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    
    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    #Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand 
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        
        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)


def get_predected_text(pred_class):
	if pred_class <= 25 :
		return chr(pred_class + 65)
	elif pred_class == 26 :
		return "space"
	else:
		return "del"

def recognize_sign_loop():
	global cam
	img = cam.read()[1]
	img = cv2.flip(img, 1)
	img = cv2.resize(img, (640, 480))

	global text
	global word
	global count_same_frame
	global num_frames
	global percentage
	old_text = text
		
	roi=img[ROI_top:ROI_bottom, ROI_right:ROI_left]
	gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

	if num_frames < 100:
		cal_accum_avg(gray_frame, accumulated_weight)
		cv2.putText(img, "DETECTING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
	else:
		hand = segment_hand(gray_frame)

		if hand is not None:
			thresh, hand_segment = hand
			cv2.drawContours(img, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)
			imgtk_thresh = ImageTk.PhotoImage(Image.fromarray(thresh))
			thresh1.imgtk = imgtk_thresh
			thresh1.configure(image=imgtk_thresh)

			thresh = cv2.resize(thresh, (40, 40))
			thresh = np.array(thresh, dtype=np.float32)
			thresh = np.reshape(thresh, (1, 40, 40, 1))

			if var.get() == 0:
				pred_probab = model.predict(thresh)[0]
				pred_class = list(pred_probab).index(max(pred_probab))
				pred_probab = max(pred_probab)
				percentage = round(pred_probab*100)
					
				if pred_probab*100 > 70:
					text = get_predected_text(pred_class)
				
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0
				
				if count_same_frame > 30:
					if text == "space":
						word = word + " "
						count_same_frame = 0
						text_area.insert(END, " ")
					elif text == "del":
						n = len(word) - 1
						word = word[:n]
						count_same_frame = 0
						text_area.delete("1.0", END)
						text_area.insert(END, word)
						#text_area.delete(str(float(n-2)), "end")
					else:
						word = word + text
						count_same_frame = 0
						text_area.insert(END, text)
			
			elif var.get() == 1:
				pred_probab = model_nb.predict(thresh)[0]
				pred_class = list(pred_probab).index(max(pred_probab))
				pred_probab = max(pred_probab)
				percentage = round(pred_probab*100)
					
				if pred_probab*100 > 70:
					text = str(pred_class)
				
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0
				
				if count_same_frame > 30:
					word = word + text
					count_same_frame = 0
					text_area.insert(END, text)
			
		else:
			text = ""
			#word = ""
			percentage = 0
			#text_area.delete("1.0", END)

	num_frames += 1

	cv2.rectangle(img, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
	percentage1['text'] = str(percentage)+"%"
	text_pred['text'] = text
	
	cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
	#img_merge = cv2.merge((r,g,b))
	imgtk = ImageTk.PhotoImage(Image.fromarray(cv2image))
	camera.imgtk = imgtk
	camera.configure(image=imgtk)
	if cam is not None:
		camera.after(10, recognize_sign_loop)

def recognize_sign():
	global cam
	if cam is None:
		cam = cv2.VideoCapture(0)
		recalcul_backgrd_btn.place(relx=0.22, rely=0.73)
		recognize_sign_loop()
	else:
		messagebox.showerror("ERROR","The Camera is already activated !! ")


def stop_rec():
	global num_frames
	global cam
	if cam is not None:
		num_frames = 0
		cam.release()
		cam = None
		recalcul_backgrd_btn.place_forget()
	else:
		messagebox.showerror("ERROR","The Camera is not activated !! ")

	img = np.full((640, 480, 3), 240, dtype=np.uint8)
	imgtk = ImageTk.PhotoImage(Image.fromarray(img))
	camera.imgtk = imgtk
	camera.configure(image=imgtk)

	thr = np.full((200, 200, 3), 240, dtype=np.uint8)
	imgtk_thresh = ImageTk.PhotoImage(Image.fromarray(thr))
	thresh1.imgtk = imgtk_thresh
	thresh1.configure(image=imgtk_thresh)

def sel():
   selection = "You selected the option " + str(var.get())

def clear_text():
	text_area.delete("1.0", END)
	global word
	word = ""

def calculing_back():
	global num_frames
	num_frames = 0


from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image

root = Tk()
root.title('ASL Translator')
width= root.winfo_screenwidth() 
height= root.winfo_screenheight()
root.geometry("%dx%d" % (width, height))

camera = Label(root)
thresh1 = Label(root)
var = IntVar()
header = Label(root, text="ASL Recognition", font=("Helvetica", 20, "bold"), height=1, width=96, bg="#FF8033")
percentageLab = Label(root, text="Percentage :", font='bold', bg="#33B3FF")
percentage1 = Label(root, text=" ", font=("Helvetica", 40), height=2, width=5, bg="white")
text_pred_lab = Label(root, text="Predected Text :", font='bold', bg="#33B3FF")
text_pred = Label(root, text=" ", font=("Helvetica", 40), height=2, width=5, bg="white")
text_lab = Label(root, text="Text :", font='bold', bg="#33B3FF")
text_area = Text(root, bg="white", height=7, width=35, font=("Helvetica", 20, "bold"))
start_btn = Button(root, text="Start the video", font="Helvetica 10 bold", height=2, width=15, bg="#33B3FF", command=lambda: recognize_sign())
stop_btn = Button(root, text="Stop the video", font="Helvetica 10 bold", height=2, width=15, bg="#FF4933", command=lambda: stop_rec())
clear_text_btn = Button(root, text="Clear the text", font="Helvetica 10 bold", height=2, width=15, bg="#FF8033", command=lambda: clear_text())
recalcul_backgrd_btn = Button(root, text="Recalculate the background", font="Helvetica 10 bold", height=2, width=25, bg="#FF8033", command=lambda: calculing_back())
radio_btn1 = Radiobutton(root, text = " Letters ", font='bold', variable = var, value = 0, command=sel)
radio_btn2 = Radiobutton(root, text = " Numbers ", font='bold', variable = var, value = 1, command=sel)
radioLab = Label(root, text="choose an option :", font='bold', bg="#33B3FF")

header.place(relx=0, rely=0)
camera.place(relx=0.01, rely=0.05)
percentageLab.place(relx=0.55, rely=0.1)
percentage1.place(relx=0.55, rely=0.15)
text_pred_lab.place(relx=0.80, rely=0.1)
text_pred.place(relx=0.80, rely=0.15)
text_lab.place(relx=0.55, rely=0.45)
text_area.place(relx=0.55, rely=0.5)
thresh1.place(relx=0.01, rely=0.65)
start_btn.place(relx=0.2, rely=0.65)
stop_btn.place(relx=0.3, rely=0.65)
radio_btn1.place(relx=0.68, rely=0.45)
radio_btn2.place(relx=0.75, rely=0.45)
radioLab.place(relx=0.55, rely=0.45)
clear_text_btn.place(relx=0.75, rely=0.8)

root.mainloop()

