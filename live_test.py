# -*- coding: utf-8 -*-

import cv2 as cv
from keras.models import model_from_json
import numpy as np
import turtle as tu

import image_preprocessing as ip
import classifier as cl

cap = cv.VideoCapture(0)
image_counter = 0

# # ['close_hand', 'no_hand', 'open_hand', 'side_hand', 'tight_hand']
# hand_position = 'test'

# load json and create model
json_file = open('./networks/Conv8Conv4Drop05Dense32Quick.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./networks/Conv8Conv4Drop05Dense32E150B32Quick.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

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

    # # Saving the image
    # cv.imwrite(f'./dataset/{hand_position}/{hand_position}_{image_counter}.jpg', frame)

    err,coeff = ip.live_preprocessing(frame)
    if(not err):
        predictions = loaded_model.predict(np.array([coeff]),verbose=0)
        print(cl.max_prediction(predictions))
    else:
        print("Error while calculating fourrier descriptors")


    # Display the resulting frame
    cv.imshow("frame", frame)

    image_counter += 1

    if cv.waitKey(1) == ord("q") or image_counter == 301:
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
