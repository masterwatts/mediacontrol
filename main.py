import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode = False,
    model_complexity = 0,
    max_num_hands = 1
    )
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # hand detected
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[0]

        for id, lm in enumerate(myHand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            
        mpDraw.draw_landmarks(img, myHand, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

    cv2.imshow("Input", img)
    cv2.waitKey(1)