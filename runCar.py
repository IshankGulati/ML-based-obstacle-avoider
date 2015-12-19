import cv2
import numpy as np
import serial
import scipy.io, scipy.special

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
weights = scipy.io.loadmat('network.mat')
theta1,theta2 = weights['theta1'], weights['theta2']
def capture():
    while(1):
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

        #edge detection
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
        #cv2.imshow('input',frame)
        img = cv2.resize(opImage,(40,30),interpolation = cv2.INTER_AREA)
        img = img.flatten()
        for j in range(len(img)):
            img[j]=(float(img[j])/255)
        img =np.array(img,dtype='float32')
        tx = predict(img,theta1,theta2)
        if tx == 4:
            ser.write("\x34")
        elif tx == 6:
            ser.write("\x36")
        elif tx ==8:
            ser.write("\x38")
        elif tx == 5:
            ser.write("\x35")
        elif tx == 2:
            ser.write("\x32")
        print tx


        k=cv2.waitKey(1)
        if k == 27: ## 27 - ASCII for escape key
            break


def sigmoid(p):
    return scipy.special.expit(p)

def predict(X,theta1,theta2):
    a1 = np.r_[np.ones((1,1)), X.reshape(X.shape[0],1)]
    z2 = theta1.dot(a1)
    a2 = sigmoid(z2)
    a2 = np.r_[np.ones((1,1)), a2]
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3)
    #the indice corresponding to maximum argument will be the prediction
    return np.argmax(z3) + 1

def main():
    output = capture()

if __name__=="__main__":
    main()

cam.release()
cv2.destroyAllWindows()

