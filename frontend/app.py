import os
import requests
import gradio as gr
import io
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

num_classes = 2
id2class = {0: 'Cat', 1: 'Dog'}

def predict_api(image_path):
    image = Image.open(image_path)
    img_name = image_path.split("/")[-1]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    url = 'http://localhost:5000/predict'  
    files = {'file_upload': (img_name, img_byte_arr, 'image/jpeg')}
    headers = {'accept': 'application/json'}

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        json_results = response.json()
        confidences = {id2class[i]: json_results["probs"][i] for i in range(num_classes)}
        return confidences, json_results
    else:
        return "Error: API request failed."

interface = gr.Interface(fn=predict_api,
                     inputs=gr.Image(
                         type='filepath', label="Upload Image",
                         height=450, width=900),
                     outputs=[
                         gr.Label(num_top_classes=2, label="Probs "),
                         gr.JSON(label="Info Output")
                     ],
                     title="Image Prediction API",
                     examples=[
                         os.path.join(os.path.dirname(__file__), "images/examples/sphynx_0.jpg"),
                         os.path.join(os.path.dirname(__file__), "images/examples/American_Eskimo_Dog_1.jpg"),
                     ],
                     description="Upload an image to get predictions from the API.")

host = os.getenv("HOST")
port = int(os.getenv("PORT"))
interface.launch(server_name=host, server_port=port)
