import cv2
import mediapipe as mp
import time
import subprocess
import math

class handDetector():
    def __init__(self, mode = False, maxHands = 2, minDetectionConfidence = .5, minTrackingConfidence = .5, modelComplexity = 0):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.modelComplexity = modelComplexity
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            model_complexity = self.modelComplexity,
            max_num_hands = self.maxHands,
            min_detection_confidence = self.minDetectionConfidence,
            min_tracking_confidence = self.minDetectionConfidence
        )
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
    
        # hand detected
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]

            if draw:
                self.mpDraw.draw_landmarks(img, myHand, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw = True):
        lmList = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 0, 150), cv2.FILLED)
        
        return lmList

def main():
    pTime = 0
    cTime = 0
    lastControl = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    status = ""
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        if (len(lmList) == 21 and time.time() - lastControl > 2):
            if (
                max([lmList[4][2], lmList[8][2], lmList[12][2], lmList[16][2], lmList[20][2]]) == lmList[20][2]
            ) and math.dist([lmList[0][1], lmList[0][2]], [lmList[12][1], lmList[12][1]]) > 20:
                # Hand is rotated
                # NEXT TRACK = LM 0 has highest x value
                # PREV TRACK = LM 0 has lowest x value
                
                # Left
                if (
                    max(lmList[0][1], lmList[4][1], lmList[8][1], lmList[12][1], lmList[16][1], lmList[20][1]) == lmList[0][1]
                ):
                    lastControl = time.time()
                    subprocess.call(['osascript', '-e', 'tell application "Spotify" to next track'])
                    status = "NEXT TRACK"
                elif (
                    min(lmList[0][1], lmList[4][1], lmList[8][1], lmList[12][1], lmList[16][1], lmList[20][1]) == lmList[0][1]
                ): # Right
                    lastControl = time.time()
                    subprocess.call(['osascript', '-e', 'tell application "Spotify" to previous track'])
                    status = "PREVIOUS TRACK"
            elif (abs(lmList[20][1] - lmList[0][1]) < 20):
                lastControl = time.time()
                status = "PLAY/PAUSE"
                subprocess.call(['osascript', '-e', 'tell application "Spotify" to playpause'])
                
        cv2.putText(img, "FPS: " + str(int(fps)), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
        
        if time.time() - lastControl < 5 and status:
            cv2.putText(img, str(status), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        cv2.imshow("Input", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()