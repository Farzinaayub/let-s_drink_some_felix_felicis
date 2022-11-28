from flask import Flask,render_template,Response
import cv2
import posemodule as pm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
detector = pm.poseDetector()
app=Flask(__name__)

def generate_frames():
    while True:
        cap=cv2.VideoCapture(0)
        rect, frame = cap.read()
        frame = detector.findPose(frame)
        lmlist = detector.findPostition(frame)
       
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +frame+ b'\r\n')
        
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')





if __name__=="__main__":
    app.run(debug=True)

