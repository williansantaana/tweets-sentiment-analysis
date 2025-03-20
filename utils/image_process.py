import io
import base64
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption_blip(base64_image: str) -> str:
    """
    Gera uma legenda para a imagem utilizando o modelo BLIP a partir de uma imagem codificada em base64.

    Parâmetros:
      base64_image (str): String em base64 representando a imagem original (em cores).

    Retorna:
      str: Legenda gerada para a imagem.
    """
    image_bytes = base64.b64decode(base64_image)
    image_buffer = io.BytesIO(image_bytes)
    image = Image.open(image_buffer).convert("RGB")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption
