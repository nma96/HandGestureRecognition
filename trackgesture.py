import cv2
import numpy as np
import os
import time
import gestureCNN as myNN

Minimum_threshold = 70

x0 = 400
y0 = 200
height = 200
width = 200

Predict_Flag = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)

counter = 0
# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = ""
mod = 0

#%%
def binaryMask(frame, x0, y0, width, height ):
    global Predict_Flag, mod, lastgesture
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    #blur = cv2.bilateralFilter(roi,9,75,75)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, Minimum_threshold, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, Minimum_threshold, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)
    
    if Predict_Flag == True:
        retgesture = myNN.predict(mod, res, frame)
        if lastgesture != retgesture :
            lastgesture = retgesture
            
            ## Checking for only PUNCH gesture here
            ## Run this app in Prediction Mode and keep Chrome browser on focus with Internet Off
            ## And have fun :) with Dino
            if lastgesture == 3:
                jump = ''' osascript -e 'tell application "System Events" to key code 49' '''
                #jump = ''' osascript -e 'tell application "System Events" to key down (49)' '''
                os.system(jump)
                print (myNN.Labels[lastgesture] + "= pause play!")

    return res

#%%
def Main():
    global Predict_Flag, mod, x0, y0, width, height, gestname, path
    
    # font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 355
    fh = 18
    
    #Call CNN model loading callback
    
    print ("Loading weight file.")
    mod = myNN.load_Neural_Network()
        
    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3,640)
    ret = cap.set(4,480)
    
    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        roi = binaryMask(frame, x0, y0, width, height)        

        ## If enabled will stop updating the main openCV windows
        ## Way to reduce some processing power :)
        cv2.imshow('Original',frame)
        cv2.imshow('ROI', roi)
        
        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff
        
        ## Use Esc key to close the program
        if key == 27:
            break
    
        ## Use g key to start gesture predictions via CNN
        elif key == ord('p'):
            Predict_Flag = not Predict_Flag
            print ("Prediction Mode - {}".format(Predict_Flag))
        
    #Realse & destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()