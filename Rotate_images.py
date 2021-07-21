import cv2, os

def flip_images():
	gest_folder = "dataseg"
	images_labels = []
	images = []
	labels = []
	for g_id in os.listdir("dataseg/"):
		if g_id != "0":	
			for i in range(999):
				path = "dataseg/"+g_id+"/"+g_id+str(i+1)+".jpg"
				new_path = "dataseg/"+g_id+"/"+g_id+str(i+1)+".jpg"
				print(path)
				img = cv2.imread(path, 0)
				img = cv2.flip(img, 1)
				cv2.imwrite(new_path, img)

flip_images()
