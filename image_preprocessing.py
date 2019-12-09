import numpy as np
import cv2 as cv
import cmath
import matplotlib.pyplot as plt

# We apply a gaussian filter to smooth the image
def gaussian_blur(img):
    sigmaX = 1.5
    sigmaY = 1.5
    ksizeX = 9
    ksizeY = 9
    return cv.GaussianBlur(img, (ksizeX,ksizeY), sigmaX=sigmaX, sigmaY=sigmaY)

# We filter the image by keeping the red if it's greater than the other components with a certain threshold
def binary_filter(img):
    seuil = 30
    width = int(img.shape[1])
    height = int(img.shape[0])
    binary = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            pixel = img[i][j]
            B = pixel[0]
            G = pixel[1]
            R = pixel[2]
            if max(R,G,B) == R and max(R,G,B)-min(R,G,B) >= seuil:
                binary[i][j] = 1
    return binary

# Now we find the contours, and we select the longest one
def findContours(binary, original):
    binary = binary.astype(np.uint8)
    contours, hierachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    max_contour_length = 0
    max_contour_index = 0
    max_contour = np.array([])
    for (index,contour) in enumerate(contours):
        if(len(contour) > max_contour_length):
            max_contour_length = len(contour)
            max_contour_index = index
            max_contour = contour
    contour_image = cv.drawContours(original, contours, max_contour_index, (0,255,0), 3)
    return max_contour, contour_image

# Compute Fourier Descriptor
def computeFourierDescriptor(contour):
    cmin = -100
    cmax = 100

    # When there is no contour
    if len(contour) == 0:
        contour = [[[0, 0]]]
          
    # We compute the Fourier coefficients
    tabcont = [complex(pt[0][0], pt[0][1]) for pt in contour]
    moyc = np.mean(tabcont)
    TC = np.fft.fft(tabcont-moyc)/len(contour)

    # We select coefficients between cmin & cmax
    coeff = np.zeros(cmax-cmin+1,dtype=complex)
    coeff[-(cmax+1):] = TC[0:cmax+1]
    coeff[0:-cmin] = TC[cmin:]

    # Phase corrections to normalize
    Phi = np.angle(coeff[((cmax-cmin)//2)-1]*coeff[((cmax-cmin)//2)+1])/2
    coeff = coeff * np.exp(np.imag(-1)*Phi)
    depha = np.angle(coeff[((cmax-cmin)//2)+1])
    for i in range(len(coeff)):
        coeff[i] = coeff[i] * np.exp(np.imag(-1)*depha*(i-((cmax-cmin)//2)))

    return coeff

# Reconstruction
def reconstructFromDescriptor(coeff):
    cmin = -100
    cmax = 100
    N = 200

    TC = np.zeros(N,dtype=complex)
    TC[0:cmax+1] = coeff[-(cmax+1):]
    TC[cmin:] = coeff[0:-cmin]
    contfil = np.fft.ifft(TC)*N
    
    # Plot reconstructed contour
    plt.plot(np.real(contfil),np.imag(contfil),'-')
    plt.show()

# Dataset Generator
def generateDataSet():
    labels = ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
    # We extract 250 images per class
    for label in labels:
        for img_index in range(250):
            img = cv.imread(f'./dataset/{label}/{label}_{img_index}.jpg', 1)
            img_blurred = gaussian_blur(img)
            img_binary = binary_filter(img_blurred)
            img_contour, img = findContours(img_binary, img)
            coeff = computeFourierDescriptor(img_contour)
            return coeff

# ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
hand_position = 'open_hand'
img = cv.imread(f'./dataset/{hand_position}/{hand_position}_0.jpg', 1)

img_blurred = gaussian_blur(img)
img_binary = binary_filter(img_blurred)
img_contour, img = findContours(img_binary, img)

cv.imshow('img', img)
cv.imshow('img_blurred', img_blurred)
cv.imshow('img_binary', img_binary)

# When everything done, release the capture
cv.waitKey(0)
cv.destroyAllWindows()