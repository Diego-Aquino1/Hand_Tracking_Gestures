import time
from tracking_hand import *

cam_width, cam_height = 640, 480

previous_time = 0
camera = cv2.VideoCapture(0)
camera.set(3, cam_width)
camera.set(4, cam_height)

detector = HandDetector(max_hands=2)

def get_gesture(raised_fingers, landmarks_list):
    # Define gestures based on the position of the fingers
    if raised_fingers == [0, 0, 0, 0, 0]:
        return "Mano cerrada"
    elif raised_fingers == [1, 1, 1, 1, 1]:
        return "Mano abierta"
    elif raised_fingers == [0, 1, 1, 0, 0]:
        return "Victoria"
    elif raised_fingers == [0, 1, 0, 0, 0] and landmarks_list[8][2] < landmarks_list[6][2]:  # index higher than the knuckle
        return "Stop"
    elif raised_fingers == [1, 0, 0, 0, 0]:
        return "Pulgar arriba"
    else:
        return "Desconocido"

while True:
    success, frame = camera.read()
    frame = detector.detect_hands(frame)
    landmarks, bounding_box = detector.get_hand_position(frame)

    if len(landmarks) != 0:
        fingers_raised = detector.check_fingers_raised()
        count_raised_fingers = fingers_raised.count(1)  # count how many fingers are raised
        gesture = get_gesture(fingers_raised, landmarks)

        # Display the number of fingers raised and the recognized gesture
        cv2.putText(frame, f'Números: {count_raised_fingers}', (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(frame, f'Gesto: {gesture}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (50, 255, 50), 3)

    cv2.imshow("Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
