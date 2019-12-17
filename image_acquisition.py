# -*- coding: utf-8 -*-

import cv2 as cv

cap = cv.VideoCapture(0)
image_counter = 251

# ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
hand_position = "test"

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

    # Resize the image by x%, can be usefull for a faster runtime
    scale_percent = 60  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)

    # Display the resulting frame
    cv.imshow("frame", resized)

    # Saving the image
    cv.imwrite(f"./dataset/{hand_position}/{hand_position}_{image_counter}.jpg", frame)
    image_counter += 1
    
    if cv.waitKey(1) == ord('q') or image_counter == 500:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
