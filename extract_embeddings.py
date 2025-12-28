import cv2
import os
import numpy as np
from imutils import paths

# -----------------------------
# BASE DIRECTORY (IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# MODEL PATHS
# -----------------------------
PROTO_PATH = os.path.join(
    BASE_DIR, "models", "deploy.prototxt"
)
CAFFE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel"
)
EMBEDDER_PATH = os.path.join(
    BASE_DIR, "models", "nn4.small2.v1.t7"
)

# -----------------------------
# VERIFY FILES EXIST
# -----------------------------
print("Checking model files...")
print("deploy.prototxt:", os.path.exists(PROTO_PATH))
print("caffemodel:", os.path.exists(CAFFE_MODEL_PATH))
print("openface model:", os.path.exists(EMBEDDER_PATH))

if not (os.path.exists(PROTO_PATH) and
        os.path.exists(CAFFE_MODEL_PATH) and
        os.path.exists(EMBEDDER_PATH)):
    raise FileNotFoundError(
        "❌ One or more model files are missing. Check the models/ folder."
    )

# -----------------------------
# LOAD MODELS
# -----------------------------
print("\nLoading face detector...")
detector = cv2.dnn.readNetFromCaffe(
    PROTO_PATH,
    CAFFE_MODEL_PATH
)

print("Loading face embedder...")
embedder = cv2.dnn.readNetFromTorch(
    EMBEDDER_PATH
)

print("✅ Models loaded successfully\n")

# -----------------------------
# DATASET PATH
# -----------------------------
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        "❌ dataset folder not found."
    )

# -----------------------------
# STORAGE
# -----------------------------
known_embeddings = []
known_names = []

image_paths = list(paths.list_images(DATASET_PATH))

print(f"Found {len(image_paths)} images in dataset\n")

# -----------------------------
# PROCESS IMAGES
# -----------------------------
for image_path in image_paths:
    print(f"Processing: {image_path}")

    name = image_path.split(os.path.sep)[-2]

    image = cv2.imread(image_path)
    if image is None:
        print("⚠️ Could not read image. Skipping.")
        continue

    (h, w) = image.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0)
    )

    detector.setInput(blob)
    detections = detector.forward()

    if detections.shape[2] == 0:
        print("⚠️ No face detected. Skipping.")
        continue

    # Pick the strongest detection
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    if confidence < 0.6:
        print("⚠️ Low confidence face. Skipping.")
        continue

    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    face = image[startY:endY, startX:endX]

    if face.size == 0:
        print("⚠️ Empty face ROI. Skipping.")
        continue

    # Create face embedding
    face_blob = cv2.dnn.blobFromImage(
        face,
        1.0 / 255,
        (96, 96),
        (0, 0, 0),
        swapRB=True,
        crop=False
    )

    embedder.setInput(face_blob)
    vec = embedder.forward()

    known_embeddings.append(vec.flatten())
    known_names.append(name)

    print(f"✅ Face encoded for: {name}\n")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
if len(known_embeddings) == 0:
    raise RuntimeError("❌ No embeddings were created. Check your dataset images.")

np.save(os.path.join(BASE_DIR, "embeddings.npy"), known_embeddings)
np.save(os.path.join(BASE_DIR, "names.npy"), known_names)

print("🎉 Embeddings saved successfully!")
print(f"Total faces encoded: {len(known_embeddings)}")


# import cv2
# import os
# import numpy as np
# from imutils import paths

# detector = cv2.dnn.readNetFromCaffe(
#     "models/deploy.prototxt",
#     "models/res10_300x300_ssd_iter_140000.caffemodel"
# )

# embedder = cv2.dnn.readNetFromTorch(
#     "models/openface.nn4.small2.v1.t7"
# )

# known_embeddings = []
# known_names = []

# image_paths = list(paths.list_images("dataset"))

# for image_path in image_paths:
#     name = image_path.split(os.path.sep)[-2]

#     image = cv2.imread(image_path)
#     (h, w) = image.shape[:2]

#     blob = cv2.dnn.blobFromImage(
#         cv2.resize(image, (300, 300)),
#         1.0, (300, 300),
#         (104.0, 177.0, 123.0)
#     )

#     detector.setInput(blob)
#     detections = detector.forward()

#     if len(detections) > 0:
#         i = np.argmax(detections[0, 0, :, 2])
#         confidence = detections[0, 0, i, 2]

#         if confidence > 0.6:
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")

#             face = image[startY:endY, startX:endX]
#             if face.size == 0:
#                 continue

#             face_blob = cv2.dnn.blobFromImage(
#                 face, 1.0/255, (96, 96),
#                 (0, 0, 0), swapRB=True, crop=False
#             )

#             embedder.setInput(face_blob)
#             vec = embedder.forward()

#             known_embeddings.append(vec.flatten())
#             known_names.append(name)

# np.save("embeddings.npy", known_embeddings)
# np.save("names.npy", known_names)

# print("✅ Embeddings extracted")
