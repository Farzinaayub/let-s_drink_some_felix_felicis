import cv2
import numpy as np
import mediapipe as mp
import time
from keras.models import load_model 
import tensorflow as tf
import numpy as np


model  = load_model('model\imageclassifier.h5')

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


    def inFrame(lst):
	    if (lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and 
                                lst[15].visibility>0.6 and lst[16].visibility>0.6):
		    return True 
	    return False


    def findPose(self,frame,draw=True):
        lst=[]
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and poseDetector.inFrame(self.results.pose_landmarks.landmark):
            for i in self.results.pose_landmarks.landmark:
                lst.append(i.x - self.results.pose_landmarks.landmark[0].x)
                lst.append(i.y - self.results.pose_landmarks.landmark[0].y)
                resize = tf.image.resize(frame, (256,256))
                yhat = model.predict(np.expand_dims(resize/255, 0))
                if yhat > 0.5: 
                    print(f'Predicted class : vrikshasana')
                    pred = 'vrikshasana'
                    cv2.putText(frame, "Vrikshasana", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)

                else:
                    print(f'Predicted class : sidhasana')
                    pred = 'sidhasana'
                    cv2.putText(frame, "Sidhasana", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)
    
                

        else: 
            cv2.putText(frame, "Make Sure Full body visible", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)
            
            #cv2.putText(frame, "no class available", (100,450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),3)
    
            
        if draw:
                self.mpDraw.draw_landmarks(frame,self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        
        return frame


    def findPostition(self,frame,draw=True):
        lmlist=[]
        if (self.results.pose_landmarks):
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = frame.shape
                #print(id,lm)
                cx,cy = int(lm.x*w),int(lm.y*h )
                
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmlist

    

#
#def main():
    #cap = cv2.VideoCapture(r'posevideos\dance.mp4')
    #pTime = 0
    #detector = poseDetector()


    #def rescale_frame(frame, percent):
        #width = int(frame.shape[1] * percent/ 100)
        #height = int(frame.shape[0] * percent/ 100)
        #dim = (width, height)
        #return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

   # while cap.isOpened():
       # rect, frame = cap.read()
       # frame = detector.findPose(frame)
      #  lmlist = detector.findPostition(frame)
      #  if len(lmlist)!=0:
       #     print(lmlist)

      #  frame = rescale_frame(frame, percent=25)

       # cTime=time.time()
       #fps=1/(cTime-pTime)
       # pTime = cTime
#
       # cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

       # cv2.imshow('image', frame)
      #  if cv2.waitKey(10) == ord('q'):
      #      break
    #cap.release()
#cv2.destroyAllWindows()
#

#if __name__ == "__main__" :
  #  main()