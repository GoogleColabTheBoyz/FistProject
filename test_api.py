from fastapi.testclient import TestClient
from PIL import Image
import io
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

    expected_keywords = ["pink", "tongue"]
    generated_description = response.json().get("translation", "").lower()
    
    for keyword in expected_keywords:
        assert keyword in generated_description

test_picture_to_text_endpoint()
