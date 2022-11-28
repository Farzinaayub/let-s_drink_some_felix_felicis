import cv2
import numpy as np
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode=False, complexity = 1, upbody=False, smooth=True, detcon=0.5, trackcon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.upbody = upbody
        self.smooth = smooth
        self.detcon = detcon
        self.trackcon = trackcon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.upbody, self.smooth, self.detcon,  self.trackcon)

    def findPose(self,frame,draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if (self.results.pose_landmarks):
            if draw:
                self.mpDraw.draw_landmarks(frame,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)

        return frame
    def findPostition(self,frame,draw=False):
        lmlist=[]
        
        lmlist.append(['vrikshasana'])
        if (self.results.pose_landmarks):
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = frame.shape
                #print(id,lm)
                cx,cy = int(lm.x*w),int(lm.y*h )
                lmlist.append([cx,cy])
            
                #else:
                   # tomerge.append([_,[cx,cy]])
            if draw:
                cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmlist

    

#
def main():
    cap = cv2.VideoCapture(r'posevideos\dance.mp4')
    pTime = 0
    detector = poseDetector()


    def rescale_frame(frame, percent):
        width = int(frame.shape[1] * percent/ 100)
        height = int(frame.shape[0] * percent/ 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

    while cap.isOpened():
        rect, frame = cap.read()
        frame = detector.findPose(frame)
        lmlist = detector.findPostition(frame)
        if len(lmlist)!=0:
            print(lmlist)

        frame = rescale_frame(frame, percent=25)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime = cTime

        cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        cv2.imshow('image', frame)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
cv2.destroyAllWindows()
#

if __name__ == "__main__" :
    main()