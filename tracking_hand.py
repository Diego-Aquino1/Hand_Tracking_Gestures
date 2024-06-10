import cv2
import mediapipe as mp
import math

class HandDetector():
    def __init__(self, detection_mode=False, max_hands=1):
        self.detection_mode = detection_mode
        self.max_hands = max_hands
        self.mp_hands_module = mp.solutions.hands
        self.hand_processor = self.mp_hands_module.Hands(
            self.detection_mode, 
            max_num_hands=self.max_hands, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.finger_tip_ids = [4, 8, 12, 16, 20]

    def detect_hands(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.process_results = self.hand_processor.process(image_rgb)

        if self.process_results.multi_hand_landmarks:
            for hand_landmarks in self.process_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands_module.HAND_CONNECTIONS)
        return image

    def get_hand_position(self, image, hand_index=0):
        x_coords = []
        y_coords = []
        bounding_box = []
        self.landmark_list = []
        if self.process_results.multi_hand_landmarks:
            target_hand = self.process_results.multi_hand_landmarks[hand_index]
            for idx, landmark in enumerate(target_hand.landmark):
                img_height, img_width, _ = image.shape
                coord_x, coord_y = int(landmark.x * img_width), int(landmark.y * img_height)
                x_coords.append(coord_x)
                y_coords.append(coord_y)
                self.landmark_list.append([idx, coord_x, coord_y])
                cv2.circle(image, (coord_x, coord_y), 5, (255, 0, 255), cv2.FILLED)

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bounding_box = x_min, y_min, x_max, y_max

            cv2.rectangle(image, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

        return self.landmark_list, bounding_box

    def check_fingers_raised(self):
        raised_fingers = []
        if self.landmark_list[self.finger_tip_ids[0]][1] > self.landmark_list[self.finger_tip_ids[0] - 1][1]:
            raised_fingers.append(1)
        else:
            raised_fingers.append(0)
        for idx in range(1, 5):
            if self.landmark_list[self.finger_tip_ids[idx]][2] < self.landmark_list[self.finger_tip_ids[idx] - 2][2]:
                raised_fingers.append(1)
            else:
                raised_fingers.append(0)
        return raised_fingers

    def measure_distance(self, point1, point2, image, circle_radius=15, line_thickness=3):
        x1, y1 = self.landmark_list[point1][1:]
        x2, y2 = self.landmark_list[point2][1:]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), line_thickness)
        cv2.circle(image, (x1, y1), circle_radius, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), circle_radius, (255, 0, 255), cv2.FILLED)
        cv2.circle(image, (center_x, center_y), circle_radius, (0, 0, 255), cv2.FILLED)
        
        distance = math.hypot(x2 - x1, y2 - y1)

        return distance, image, [x1, y1, x2, y2, center_x, center_y]