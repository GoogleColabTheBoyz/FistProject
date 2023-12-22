from fastapi.testclient import TestClient
from PIL import Image
import io
import requests
from main import app

client = TestClient(app)

def test_picture_to_text_endpoint():
    image_path = "public/img/1.jpg"
    image = Image.open(image_path)
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes.seek(0)

    response = client.post("/picturetotext", files={"file": ("image.jpg", image_bytes, "image/jpeg")})
    assert response.status_code == 200
    assert "translation" in response.json()
test_picture_to_text_endpoint()
