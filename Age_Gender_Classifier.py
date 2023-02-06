import cv2 as cv

def faceBox(facenet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    facenet.setInput(blob)
    detection = facenet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)

            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)

            bboxs.append([x1,y1,x2,y2])
            cv.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

    return frame, bboxs

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

facenet = cv.dnn.readNet(faceModel, faceProto)
agenet = cv.dnn.readNet(ageModel, ageProto)
gendernet = cv.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

capture = cv.VideoCapture(0)
padding = 20

while True:
    ret, frame = capture.read()
    frame, bboxs = faceBox(facenet, frame)
    for bbox in bboxs:
        # face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        gendernet.setInput(blob)
        gendePred = gendernet.forward()
        gender = genderList[gendePred[0].argmax()]

        agenet.setInput(blob)
        agePred = agenet.forward()
        age = ageList[agePred[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv.rectangle(frame, (bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0), -1)
        cv.putText(frame, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)


    cv.imshow("Age & Gender classifier", frame)

    if cv.waitKey(20) & 0xFF == ord(' '):
        break

capture.release()
cv.destroyAllWindows()