import pytesseract as tess
from PIL import Image
import pyttsx3
tess.pytesseract.tesseract_cmd=r'C:\Users\Pawan Kumar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
print("Hi")
img=Image.open('img2.jpg')
text=tess.image_to_string(img)
print(text)
text_speech=pyttsx3.init()
text_speech.say(text)
text_speech.runAndWait()