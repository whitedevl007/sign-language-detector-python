
import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Try different camera indices until you find the correct one
cap = cv2.VideoCapture(0)  # Change the index here

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        
        # Check if frame size is valid
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('frame', frame)
        else:
            print("Error: Invalid frame size")
            break

        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame")
            break
        
        # Check if frame size is valid
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('frame', frame)
        else:
            print("Error: Invalid frame size")
            break
        
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()
