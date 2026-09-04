import json
from io import BytesIO

import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "model.savedmodel"
LABELS_PATH = "labels.txt"

IMAGE_SIZE = (224, 224)


# ============================================================
# Application
# ============================================================

app = FastAPI()


# ============================================================
# Load Model
# ============================================================

print("Loading TensorFlow SavedModel...")

model = tf.saved_model.load(
    MODEL_PATH
)

infer = model.signatures[
    "serving_default"
]

print("Model loaded successfully.")

# ============================================================
# Model Signature
# ============================================================

print("Input signature:")
print(infer.structured_input_signature)

print("Output signature:")
print(infer.structured_outputs)


# ============================================================
# Load Labels
# ============================================================

def load_labels():

    labels = []

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                " ",
                1
            )

            if len(parts) == 2:
                labels.append(
                    parts[1]
                )
            else:
                labels.append(
                    line
                )

    return labels


labels = load_labels()


print("Loaded labels:")

for index, label in enumerate(labels):

    print(
        f"{index}: {label}"
    )


# ============================================================
# Prediction
# ============================================================

def predict_image(image_bytes):

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")


    # --------------------------------------------------------
    # Resize and center crop
    #
    # Matches Teachable Machine:
    #
    # ImageOps.fit(
    #     image,
    #     (224, 224),
    #     Image.Resampling.LANCZOS
    # )
    # --------------------------------------------------------

    image = ImageOps.fit(
        image,
        IMAGE_SIZE,
        Image.Resampling.LANCZOS
    )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    )


    # --------------------------------------------------------
    # Normalize
    #
    # 0   -> -1
    # 127 -> approximately 0
    # 255 -> 1
    # --------------------------------------------------------

    image_array = (
        image_array / 127.5
    ) - 1.0


    # --------------------------------------------------------
    # Add batch dimension
    #
    # [224, 224, 3]
    #        ↓
    # [1, 224, 224, 3]
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Run TensorFlow inference
    # --------------------------------------------------------

    prediction = infer(
        sequential_1_input=tf.convert_to_tensor(
            image_array,
            dtype=tf.float32
        )
    )


    # --------------------------------------------------------
    # Get prediction output
    # --------------------------------------------------------

    predictions = prediction[
        "sequential_3"
    ].numpy()[0]


    # --------------------------------------------------------
    # Find highest prediction
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(
            predictions
        )
    )

    confidence = float(
        predictions[predicted_index]
    )


    # --------------------------------------------------------
    # Get label
    # --------------------------------------------------------

    predicted_label = labels[
        predicted_index
    ]


    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "success": True,
        "label": predicted_label,
        "confidence": confidence
    }

# ============================================================
# HTTP
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "image-recognition-server"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    print(
        "Client connected."
    )

    try:

        while True:

            # ------------------------------------------------
            # Receive image
            # ------------------------------------------------

            image_bytes = (
                await websocket.receive_bytes()
            )


            print(
                f"\nImage received: "
                f"{len(image_bytes)} bytes"
            )


            # ------------------------------------------------
            # Run prediction
            # ------------------------------------------------

            try:

                result = predict_image(
                    image_bytes
                )


                # --------------------------------------------
                # Send prediction to client
                # --------------------------------------------

                await websocket.send_text(
                    json.dumps(
                        result
                    )
                )


                print(
                    f"Prediction: "
                    f"{result['label']} "
                    f"({result['confidence']:.2%})"
                )


            except Exception as exception:

                print(
                    f"Prediction error: "
                    f"{exception}"
                )


                error_response = {
                    "success": False,
                    "error": str(
                        exception
                    )
                }


                await websocket.send_text(
                    json.dumps(
                        error_response
                    )
                )


    except WebSocketDisconnect:

        print(
            "Client disconnected."
        )


