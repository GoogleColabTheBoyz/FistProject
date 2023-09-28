from transformers import pipeline
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel
from IPython.display import Audio

classifier = pipeline("sentiment-analysis", "blanchefort/rubert-base-cased-sentiment")

print(classifier("Ответ убил меня. Я стану отцом двухсотый раз!"))
print(classifier("Ответ убил меня. Я стану отцом двухсотый раз!"))

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)

img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

ImageToTextInput = processor.decode(out[0], skip_special_tokens=True)

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=[ImageToTextInput],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)

sampling_rate = model.generation_config.sample_rate
Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)
