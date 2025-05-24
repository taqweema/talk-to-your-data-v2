from PIL import Image
import pytesseract

def extract_text_from_image(image_file):
    """
    Takes an uploaded image file and returns extracted text using Tesseract OCR.
    Supports .jpg and .png formats.
    """
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text
