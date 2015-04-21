import numpy as np
import scipy.special,scipy.optimize,scipy.io,scipy.misc
import PIL.Image
import cv2

def load():
    entries = 3404
    inputs = []
    outputs = scipy.genfromtxt("controls.txt",delimiter=" ")
    m=len(outputs)
    outputs = outputs.reshape(m,1)
    outputs = np.array(outputs,dtype = 'int64')

    for i in range(1,entries+1):
        img = cv2.imread("testImages\\test"+str(i)+".jpg",-1)
        img = img.flatten()
        #img = img.tolist()
        for j in range(len(img)):
            img[j]=(float(img[j])/255)
        img =np.array(img,dtype='float32')
        inputs.append(img)
    inputs = np.array(inputs,dtype = 'float32')
    trainingData = (inputs,outputs)
    print "Load successful"
    return trainingData

def main():
	d = load()
	
if __name__ == '__main__':
	main()
