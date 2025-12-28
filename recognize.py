import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

detector = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

embedder = cv2.dnn.readNetFromTorch(
    "models/nn4.small2.v1.t7"
)

known_embeddings = np.load("embeddings.npy")
known_names = np.load("names.npy")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    detector.setInput(blob)
    detections = detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            face_blob = cv2.dnn.blobFromImage(
                face, 1.0/255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False
            )

            embedder.setInput(face_blob)
            vec = embedder.forward().flatten()

            similarities = cosine_similarity([vec], known_embeddings)[0]
            best_match = np.argmax(similarities)

            if similarities[best_match] > 0.7:
                name = known_names[best_match]
            else:
                name = "Unknown"

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            cv2.putText(frame, name, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Accurate Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
