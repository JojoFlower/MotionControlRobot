import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

cap = cv.VideoCapture(0)
seuil=10
cmin = -10
cmax = 10
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
    
    # Apply gaussian blur to the image, other preprocessing could be necessary
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

    #Try to use channel substraction instead
    B_channel,G_channel,R_channel = cv.split(blurred)
    R_over_B = R_channel-np.add(seuil,B_channel)
    R_over_G = R_channel-np.add(seuil,G_channel)
    R_over_G_and_B = np.add(R_over_B,R_over_G)
    ret,thresh1 = cv.threshold(R_over_G_and_B,100,255,cv.THRESH_BINARY_INV)

    #Otherwise we can also use inRange which applies a threshold on each channel
    binaire = np.zeros((height,width))
    binary = cv.inRange(blurred,(0,0,130),(175,175,255))

    #Now we find the contours, and we select the longest one
    contours,hierachy = cv.findContours(thresh1,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    max_contour_length = 0
    max_contour_index = 0
    max_contour = np.array([])
    for (index,contour) in enumerate(contours):
        if(len(contour) > max_contour_length):
            max_contour_length = len(contour)
            max_contour_index = index
            max_contour = contour
    print("Contour number : {}, length: {}".format(max_contour_index,max_contour_length))
    contour_image = cv.drawContours(resized, contours, max_contour_index, (0,255,0), 3)
    
    #This next section calculates the fourrier descriptors
    #First we nedd to get the fft after normalizing
    moyc = np.mean(max_contour)
    normalized_contour = max_contour - moyc / max_contour_length
    TC = np.fft.fft(normalized_contour)
    TC = np.reshape(TC,(max_contour_length,2))
    
    #Now select only the coefficients of interest between cmin and cmax
    coeff = np.zeros((cmax-cmin+1,2))
    coeff[-(cmax+1):] = TC[0:cmax+1]
    coeff[0:-cmin] = TC[cmin:]

    #Phase correction in order to normalize
    Phi = np.angle(coeff[((cmax-cmin)//2)-1]*coeff[((cmax-cmin)//2)+1])/2
    coeff = coeff * np.exp(np.imag(-1)*Phi)
    
    # depha = np.angle(coeff[((cmax-cmin)//2)+1])
    # coeff = coeff * np.exp(np.imag(-1)*)
    
    # Display the resulting frame
    cv.imshow('frame', resized)
    cv.imshow('binaire', binary)
    cv.imshow('1', thresh1)
    cv.imshow('R_G', R_over_G)
    cv.imshow('R_B', R_over_B)
    cv.imshow('R_G_B', R_over_G_and_B)


    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()