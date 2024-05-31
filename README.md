
<div align="center">
  <h1>Gesture Volume Control Using Python</h1>
  <img alt="output" src="images/output.gif" />
 </div>

> This Project uses Python and some modules to control System Volume 

##  REQUIREMENTS
+ opencv-python
+ mediapipe
+ comtypes
+ numpy
+ pycaw

```bash
pip install -r requirements.txt
```
***
### OPENCV-PYTHON

OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides a common infrastructure for computer vision applications and accelerates the use of machine perception in commercial products. OpenCV was designed for computational efficiency and with a strong focus on real-time applications.

Key Features of OpenCV:
>Image Processing,Video Processing,Object Detection and Tracking, 2D Features Framework.


### MEDIAPIPE

Mediapipe is an open-source framework developed by Google that provides a comprehensive set of tools and solutions for building perception pipelines. It is designed to facilitate the development of machine learning applications that process live and static media data, such as videos and images. Mediapipe is particularly powerful for tasks that involve real-time processing on mobile devices, web, and desktops.

Key Features of Mediapipe:
>Initialize Mediapipe Hands Solution, Process Frames, Volume Control Logic and Display Results.


### COMTYPES

comtypes is a pure Python library that allows for the creation and management of COM (Component Object Model) objects. It provides a way to interact with COM interfaces and automate tasks in Windows applications. This is particularly useful for controlling system-level features like audio volume, which is where comtypes is used in conjunction with the pycaw (Python Core Audio Windows) library to control system audio settings.

Key Features of Comtypes:
>Initialize Audio Interface, Capture Video and Detect Hand Gestures and Map Gestures to Volume Control.


### NUMPY

NumPy (Numerical Python) is a fundamental package for scientific computing in Python. It provides support for arrays, matrices, and a large collection of mathematical functions to operate on these data structures efficiently. In the context of a hand gesture volume controller, NumPy is used to perform mathematical operations that facilitate the detection and interpretation of hand gestures.

Key Features of Numpy:
>Calculate Distances and Map Distances to Volume Levels.


### PYCAW

pycaw (Python Core Audio Windows) is a Python library that provides access to Windows Core Audio APIs, allowing you to control audio settings on Windows. It is especially useful for tasks like adjusting the system volume, muting audio, and retrieving audio device information. In the context of a hand gesture volume controller, pycaw enables you to change the system volume based on hand gestures detected by a webcam.

Key Features of Pycaw:
>Access Audio Devices and Integration with Other Libraries.





<div align="center">
    <img alt="mediapipeLogo" src="Images/hand_landmarks.png" height="200 x" />
    <img alt="mediapipeLogo" src="images/htm.jpg" height="360 x" weight ="640 x" />
    
</div>


<br>



## 📝 CODE EXPLANATION
<b>Importing Libraries</b>
```py
import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
```
***
Solution APIs 
```py
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
```
***

Volume Control Library Usage 
```py
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
```
***
Getting Volume Range using `volume.GetVolumeRange()` Method
```py
volRange = volume.GetVolumeRange()
minVol , maxVol , volBar, volPer= volRange[0] , volRange[1], 400, 0
```
***
Setting up webCam using OpenCV
```py
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3,wCam)
cam.set(4,hCam)
```
***
Using MediaPipe Hand Landmark Model for identifying Hands 
```py
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cam.isOpened():
    success, image = cam.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )
```
***
Using multi_hand_landmarks method for Finding postion of Hand landmarks
```py
lmList = []
    if results.multi_hand_landmarks:
      myHand = results.multi_hand_landmarks[0]
      for id, lm in enumerate(myHand.landmark):
        h, w, c = image.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])    
```
***
Assigning variables for Thumb and Index finger position
```py
if len(lmList) != 0:
      x1, y1 = lmList[4][1], lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]
```
***
Marking Thumb and Index finger using `cv2.circle()` and Drawing a line between them using `cv2.line()`
```py
cv2.circle(image, (x1,y1),15,(255,255,255))  
cv2.circle(image, (x2,y2),15,(255,255,255))  
cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
length = math.hypot(x2-x1,y2-y1)
if length < 50:
    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
```
***
Converting Length range into Volume range using `numpy.interp()`
```py
vol = np.interp(length, [50, 220], [minVol, maxVol])
```
***
Changing System Volume using `volume.SetMasterVolumeLevel()` method
```py
volume.SetMasterVolumeLevel(vol, None)
volBar = np.interp(length, [50, 220], [400, 150])
volPer = np.interp(length, [50, 220], [0, 100])
```
***
Drawing Volume Bar using `cv2.rectangle()` method
```py
cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
        1, (0, 0, 0), 3)}

```
***
Displaying Output using `cv2.imshow` method
```py
cv2.imshow('handDetector', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
```
***
Closing webCam
```py
cam.release()
```
***

<div align = "center">
<h2>📬 Contact</h2>

If you want to contact me, you can reach me through below handles.

<a href="https://twitter.com/prrthamm"><img src="https://upload.wikimedia.org/wikipedia/fr/thumb/c/c8/Twitter_Bird.svg/1200px-Twitter_Bird.svg.png" width="25">@prrthamm</img></a>&nbsp;&nbsp; <a href="https://www.linkedin.com/in/pratham-bhatnagar/"><img src="https://www.felberpr.com/wp-content/uploads/linkedin-logo.png" width="25"> Pratham Bhatnagar</img></a>

</div>
