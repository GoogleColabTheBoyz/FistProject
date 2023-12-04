from transformers import pipeline
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel
import streamlit as st
import soundfile as sf
import io
from io import BytesIO
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
from pydantic import BaseModel
import numpy as np
import requests


@st.cache_resource
def load_model():
    return BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large"
    )


@st.cache_resource
def load_processor():
    return BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")


class Item(BaseModel):
    text: str


app = FastAPI()

processor = load_processor()
model = load_model()

TEXT = "a potography of"


@app.get("/")
async def root():
    return {"message": "I like pizza"}


# this function translates accepts image and turns it into text
@app.post("/picturetotext")
async def pictureToText(file: UploadFile = File(...)):
    raw_image = Image.open(file.file).convert("RGB")
    inputs = processor(raw_image, TEXT, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)


# this function translates accepts text and turns it into audio
# @app.post("/texttovoice")
# async def textToVoice(item: Item):
# inputs = processor2(
# text=[item.text],
# return_tensors="pt",
# )
# speech_values = model2.generate(**inputs, do_sample=True)
# sampling_rate = model2.generation_config.sample_rate
# audio_data = speech_values.cpu().numpy().squeeze()
# audio_buffer = io.BytesIO()
# sf.write(audio_buffer, audio_data, sampling_rate, format="WAV")
# audio_buffer.seek(0)
# return audio_buffer


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

#
# if st.button("translate to audio"):
#
# processor = AutoProcessor.from_pretrained("suno/bark-small")
# model = AutoModel.from_pretrained("suno/bark-small")
#
# inputs = processor(
# text=[ImageToTextInput],
# return_tensors="pt",
# )
#
# speech_values = model.generate(**inputs, do_sample=True)
#
# sampling_rate = model.generation_config.sample_rate
# audio_data = speech_values.cpu().numpy().squeeze()
# audio_buffer = io.BytesIO()
# sf.write(audio_buffer, audio_data, sampling_rate, format='WAV')
# audio_buffer.seek(0)
#
# st.audio(audio_buffer, format='audio/wav')
#
