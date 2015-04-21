import cv2
import numpy as np
import time
import serial

cam = cv2.VideoCapture(1)

cam.set(3,160)
cam.set(4,120)

ser = serial.Serial(
    port='COM7',
    baudrate=9600,
    timeout=0,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)

def capture():
    var=3405
    a='5 ' 
    f=open('trainingData/controls.txt','w')
    k=53
    while var > 0:

        minRed = 127
        maxRed = 127
        minRedGreenRange = 127
        maxRedGreenRange = 127
        minRedBlueRange = 127
        maxRedBlueRange = 127

        minLineLength = 80
        maxLineGap = 20
        ret,frame = cam.read()
        height, width, channels = frame.shape
        opImage = np.zeros((height,width,1),np.uint8)

        #finding threshols for image segmentation
        for i in range(100,height):
            for j in range(70,width-70):
                blue = frame[i,j,0]
                green = frame[i,j,1]
                red = frame[i,j,2]

                if blue==0 and red==0 and green==0:
                    continue
                
                if red < minRed:
                    minRed = red
                    
                if red > maxRed:
                    maxRed = red
                    
                redGreenRange = int(red)-int(green)
                redBlueRange = int(red)-int(blue)
                
                if redGreenRange < minRedGreenRange:
                    minRedGreenRange = redGreenRange

                if redGreenRange > maxRedGreenRange:
                    maxRedGreenRange = redGreenRange

                if redBlueRange < minRedBlueRange:
                    minRedBlueRange = redBlueRange

                if redBlueRange > maxRedBlueRange:
                    maxRedBlueRange = redBlueRange

        #colour segmentation based on calculated thresholds
        for i in range(height):
            for j in range(width):
                blue = frame[i,j,0]
                green = frame[i,j,1]
                red = frame[i,j,2]
                
                redGreen = int(red)-int(green)
                redBlue = int(red)-int(blue)

                if red > minRed and red < maxRed and redGreen > minRedGreenRange and redGreen < maxRedGreenRange and redBlue > minRedBlueRange and redBlue < maxRedBlueRange:
                    opImage[i,j] = 255
                else:
                    opImage[i,j] = 0
                    
        #line segmentation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, minLineLength, maxLineGap)
        
        if lines != None:
            for i in range(len(lines[0])):
                points=[0 for j in range(4)]

                if lines[0,i,0] < lines[0,i,2]:
                    points[0] = (lines[0,i,0],0)
                    points[1] = (lines[0,i,2],0)
                    points[2] = (lines[0,i,2],lines[0,i,3])
                    points[3] = (lines[0,i,0],lines[0,i,1])
                else:
                    points[0] = (lines[0,i,2],0)
                    points[1] = (lines[0,i,0],0)
                    points[2] = (lines[0,i,0],lines[0,i,1])
                    points[3] = (lines[0,i,2],lines[0,i,3])
                polyPoints = np.array([tuple(i) for i in points],np.int32)
                cv2.fillConvexPoly(opImage, polyPoints, [0,0,0])

        cv2.imshow('output',opImage)
        cv2.imshow('input',frame)
        k=cv2.waitKey(1)
        if k!= -1:
            if k == 53 :
                ser.write("\x35")
            else:
                resOutput = cv2.resize(opImage,(40,30),interpolation = cv2.INTER_AREA)
                cv2.imwrite('testImages/test'+str(var)+'.jpg',resOutput)
                cv2.imwrite('originalImages/original'+str(var)+'.jpg',frame)     
                var = var+1
                if k == 27: ## 27 - ASCII for escape key
                    break        
                if k == 52:
                    ser.write("\x34")
                    a='4 '
                elif k == 56:
                    a='8 '
                    ser.write("\x38")
                elif k == 50:
                    a='2 '
                    ser.write("\x32")
                elif k == 54:
                    a='6 '
                    ser.write("\x36")
                f.write(a)



#wait for the two threads to complete
if __name__=="__main__":
    capture()

cam.release()
cv2.destroyAllWindows()
    
