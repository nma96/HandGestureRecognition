import cv2
import time

x0 = 400
y0 = 200
height = 200
width = 200

minValue = 70

counter = 0

cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	frame = cv2.flip(frame, 3)

	cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
	roi = frame[y0:y0+height, x0:x0+width]
	
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),2)
	
	th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	cv2.imshow('Original',frame)
	cv2.imshow('gray',gray)
	cv2.imshow('blur',blur)
	cv2.imshow('th3',th3)
	cv2.imshow('frame',res)

	key = cv2.waitKey(10) & 0xff

	if key == ord('s'):
		counter = counter + 1
		name = "img" + str(counter)
		print("Saving img:"),name
		cv2.imwrite(name + ".png", res)
		time.sleep(0.04)

	elif key == 27 :
		break

cap.release()
cv2.destroyAllWindows()