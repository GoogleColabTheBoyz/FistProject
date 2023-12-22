from PIL import Image
from torchvision import transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from pydantic import BaseModel
import requests
import streamlit as st

app = FastAPI()
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
TEXT = "a photography of"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class Item(BaseModel):
    text: str

@app.post("/picturetotext")
async def pictureToText(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    image = transform(image)
    inputs = processor(image, TEXT, return_tensors="pt")
    out = model.generate(**inputs, max_length=50)  # Установите значение max_length в нужное вам число
    return {"translation": processor.decode(out[0], skip_special_tokens=True)}

st.title("From image to audio")
st.header("This neural network voices what is depicted in your image")
st.subheader("To try it, upload your image by clicking on the button below")

uploadImage = st.file_uploader("Choose image")

if uploadImage is not None:
    bytes_data = uploadImage.read()
    bytes_io = BytesIO(bytes_data)
    bytes_io.seek(0)
    res = requests.post("http://127.0.0.1:8001/picturetotext", files={"file": bytes_io})
    st.text(res.json())
else:
    st.error("Upload image!")
