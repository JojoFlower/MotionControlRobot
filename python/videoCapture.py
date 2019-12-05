import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
seuil=30
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    #Resize the image by x%, can be usefull for a faster runtime
    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA) 

    # Our operations on the frame come here
    
    binaire = np.zeros((height,width))
    blurred = cv.GaussianBlur(resized, (7,7), sigmaX=1.5, sigmaY=1.5)
    
    #Cannot use original thresholding because hand made python loops are too slow
    # for i in range(blurred.shape[0]):
    #     for j in range(blurred.shape[1]):
    #         pixel = blurred[i][j]
    #         B = pixel[0]
    #         G = pixel[1]
    #         R = pixel[2]
    #         if(((R-B)>=seuil) or ((R-G)>=seuil)):
    #             binaire[i][j] = 1

    #Instead we use inRange which applies a threshold on each channel
    binary = cv.inRange(blurred,(0,0,130),(175,175,255))

    #Now we find the contours, and we select the longest one
    contours,hierachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    max_contour_length = 0
    max_contour_index = 0
    for (index,contour) in enumerate(contours):
        if(len(contour) > max_contour_length):
            max_contour_length = len(contour)
            max_contour_index = index
    print("Contour number : {}, length: {}".format(max_contour_index,max_contour_length))
    contour_image = cv.drawContours(resized, contours, max_contour_index, (0,255,0), 3)
    
    # Display the resulting frame
    cv.imshow('frame', resized)
    cv.imshow('blurred', blurred)
    cv.imshow('binaire', binary)
    cv.imshow('contour', contour_image)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()