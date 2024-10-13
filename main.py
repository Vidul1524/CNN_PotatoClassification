from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO  # create an in-memory file that stores binary data (like bytes) instead of using a real file on disk
from PIL import Image   # used to read images in python
import tensorflow as tf

app = FastAPI()   # instance/object of FastAPI

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows all origins; you can restrict it to your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variable 'MODEL' which stores our ml model (created in collab)
MODEL = tf.keras.models.load_model("../saved_models/model1/model1.keras")

# class names (outputs):
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


# 'async' allows server to handle multiple requests at a time
# coroutines: async def   --- this is perfomed using 'await' keyword
# Just to check if our sever is alive
@app.get("/ping")   # link is ----> http://localhost:8000/ping
async def ping():
    return "Hello, I am alive"


# This function is used to convert our uploaded image into numpy array/tensor so that our model can predict
def read_file_as_image(data) -> np.ndarray:   # returning np.ndarray as output
    image = Image.open(BytesIO(data)).resize((256, 256))  # Use model input size
    img_array = np.array(image)  # converting to numpy array
    return img_array / 255.0  # Normalize



# Now this is post method (to send) with function name: predict
# We will have 'upload file' option on our site so will use UploadFile (already inbuilt in fastapi)
# Means whoever is calling 'predict' link -- they will upload a file and get the prediction

@app.post("/predict")
async def predict(file: UploadFile = File(...)):   # from inbuilt FastAPI UploadFile
    # Now we have a file uploaded (image) --- we need to convert it into numpy array or tensor
    # so that our model can predict it
    # reading the image and converting:

    image = await file.read()  # await to use async def

    # Convert the image to a format compatible with the model
    img = Image.open(BytesIO(image)).resize((256, 256))  # Adjust size if needed
    img_array = np.array(img) / 255.0  # Normalize image

    # our image is [256,256,3] but our model takes [[256,256,3]]
    img_batch = np.expand_dims(img_array, axis=0)  # increasing one dimension

    # Predict the class
    predictions = MODEL.predict(img_batch)  # predicting the image using our ml model  ---> eg: [[23123,123132,13123]]

    # predictions[0] coz it gives 2d array so write [0] also
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  # final prediction value (max value among three)
    confidence = np.max(predictions[0])  # confidence value

    return {"class": predicted_class, "confidence": confidence}

# Running through uvicorn and not through command
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


# You can also check each method and function one by one using---
# http://localhost:8000/docs
