import cv2,time
import numpy as np
import pandas,datetime,matplotlib.pyplot as plt

def ROI(imag,px1,py1,px2,py2,px3,py3):
	height=imag.shape[0]
	polygons=np.array([[(px1,py1),(px2,py2),(px3,py3)]])
	mask=np.zeros_like(imag)
	cv2.fillPoly(mask,polygons,255)
	masked_img=cv2.bitwise_and(imag,mask)
	return masked_img

cap=cv2.VideoCapture("./cars.mp4")
f_frame=None
frame_number=0
f=0

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)

while cap.isOpened():

	_,frame=cap.read()
	frame=cv2.resize(frame,(640,400))
	g_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gaus_img=cv2.GaussianBlur(g_img,(21,21),0)
	FrameRoi=ROI(gaus_img,640,398,640,125,0,200)
	#plt.imshow(FrameRoi)
	#plt.show()
	if f_frame is None:
		f_frame=gaus_img
	f_FrameRoi=ROI(f_frame,640,398,640,125,0,200)
	#plt.imshow(f_FrameRoi)
	#plt.show()	
	delta_f=cv2.absdiff(f_FrameRoi,FrameRoi)
	thresh_delta=cv2.threshold(delta_f,30,255,cv2.THRESH_BINARY)[1]
	thresh_delta=cv2.dilate(thresh_delta,None,iterations=0)
	(cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour)<2500:
			continue
		
		(x,y,w,h)=cv2.boundingRect(contour)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

	cv2.imshow("Treshold",thresh_delta)
	cv2.imshow("Prueba",frame)

	if cv2.waitKey(1)==ord("q") or f==560:
		break
	f+=1

print(f)
cap.release()
cv2.destroyAllWindows()




