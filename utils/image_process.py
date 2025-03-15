import cv2
import io
import numpy as np
import base64
import torch
import pytesseract
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def preprocess_image(base64_image):
    # Decodificar a string base64 para bytes
    image_bytes = base64.b64decode(base64_image)
    
    # Converter os bytes em um array numpy
    np_array = np.frombuffer(image_bytes, np.uint8)
    
    # Decodificar o array para uma imagem colorida
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem. Verifique se o base64 está correto.")
    
    # Redimensionar a imagem se ela for muito grande (opcional)
    max_dim = 1000  # ajuste esse valor conforme necessário
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scaling_factor = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desfoque gaussiano para reduzir ruídos
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar limiarização adaptativa para realçar contrastes
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # Aplicar uma operação morfológica para eliminar pequenos ruídos residuais
    kernel = np.ones((3, 3), np.uint8)
    processed_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return processed_img


def extract_text_from_image(processed_img):
    """
    Aplica OCR na imagem pré-processada e retorna o texto extraído.
    
    Parâmetros:
      processed_img (np.ndarray): Imagem pré-processada em escala de cinza ou binarizada.
      
    Retorna:
      str: Texto extraído da imagem.
    """
    # Configuração customizada para o Tesseract
    # --oem 3: Usa o mecanismo de OCR padrão (baseado em redes neurais)
    # --psm 6: Assume uma única caixa de texto uniforme
    custom_config = r'--oem 3 --psm 6'
    
    # Extração do texto (para português, defina lang='por'; ajuste conforme necessário)
    text = pytesseract.image_to_string(processed_img, lang='por', config=custom_config)
    return text

def generate_caption_blip(base64_image: str) -> str:
    """
    Gera uma legenda para a imagem utilizando o modelo BLIP a partir de uma imagem codificada em base64.

    Parâmetros:
      base64_image (str): String em base64 representando a imagem original (em cores).

    Retorna:
      str: Legenda gerada para a imagem.
    """
    # Decodificar a string base64 para bytes
    image_bytes = base64.b64decode(base64_image)
    
    # Criar um objeto de BytesIO para abrir a imagem
    image_buffer = io.BytesIO(image_bytes)
    
    # Abrir a imagem usando o PIL e convertê-la para o formato RGB
    image = Image.open(image_buffer).convert("RGB")
    
    # Carregar o processador e o modelo BLIP
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Preparar a entrada para o modelo
    inputs = processor(image, return_tensors="pt")
    
    # Gerar a legenda da imagem
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption
