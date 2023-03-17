# Face Recognition and Liveness Detection Module with API

- RetinaNet is Used for Face Detection Backbone
- ArcFace Obtain Highly Discriminative Features for Face Recognition
- Liveness Detection Utilize Eye Blink Counter and Head Position Detection

Following Application is Available:
- Tkinter Windows App  
- Flask Web App


![Postman API Call](Postman.png)

---------

request ('/face' PostAPI): 
- under 5MB .mp4 video file
- valid national code

response ('/face' PostAPI):
- code: 20
    - Liveness and Verification Passed
- code: 41
    - Verification Failed
- code: 42
    - Liveness Failed
- code: 40
    - Unknown Failure in Face Module
