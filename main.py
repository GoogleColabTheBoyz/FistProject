from transformers import pipeline
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel
import streamlit as st
import soundfile as sf
import io

st.title('From image to audio')
st.header('This neural network voices what is depicted in your image')
st.subheader('To try it, upload your image by clicking on the button below')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
TEXT = "a potography of"

uploadImage = st.file_uploader("Choose image")


if uploadImage is not None:


    raw_image = Image.open(uploadImage).convert("RGB")

    inputs = processor(raw_image, TEXT, return_tensors="pt")

    out = model.generate(**inputs)
    st.text(processor.decode(out[0], skip_special_tokens=True))

    ImageToTextInput = processor.decode(out[0], skip_special_tokens=True)
else:
    st.error("Upload image!")




if st.button("translate to audio"):

    processor = AutoProcessor.from_pretrained("suno/bark-small")
    model = AutoModel.from_pretrained("suno/bark-small")

    inputs = processor(
        text=[ImageToTextInput],
        return_tensors="pt",
    )

    speech_values = model.generate(**inputs, do_sample=True)

    sampling_rate = model.generation_config.sample_rate
    audio_data = speech_values.cpu().numpy().squeeze()
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, sampling_rate, format='WAV')
    audio_buffer.seek(0)

    st.audio(audio_buffer, format='audio/wav')
