import cv2
import numpy as np
import glob

gridsize = 8
clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(gridsize,gridsize))
frames = glob.glob("*.jpg")
for frame in frames:
	bgr = cv2.imread(frame)
	bgr_shape = bgr.shape 
	out_bgr = np.zeros(bgr_shape)
	out_bgr_clahe = np.zeros(bgr_shape)
	equalized_red = cv2.equalizeHist(bgr[:,:,2])
	equalized_blue = cv2.equalizeHist(bgr[:,:,0])
	equalized_green = cv2.equalizeHist(bgr[:,:,1])
	clahe_red = clahe.apply(equalized_red)
	clahe_green = clahe.apply(equalized_green)
	clahe_blue = clahe.apply(equalized_blue)
	out_bgr[:,:,0] = equalized_blue
	out_bgr[:,:,1] = equalized_green
	out_bgr[:,:,2] = equalized_red
	out_bgr_clahe[:,:,0] = clahe_blue
	out_bgr_clahe[:,:,1] = clahe_green
	out_bgr_clahe[:,:,2] = clahe_red

	cv2.imwrite("Eq/"+frame,out_bgr)
	cv2.imwrite("Eq_CLAHE/"+frame,out_bgr_clahe)
	# cv2.imwrite("Eq_lab/"+frame,out_bgr_clahe_hsv)