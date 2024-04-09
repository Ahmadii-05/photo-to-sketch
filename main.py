import numpy as np
import cv2
from flask import Flask, render_template, request
from io import BytesIO
import base64

app = Flask(__name__)

def convert_to_sketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray_image = 255 - gray_image
    blurred_img = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    inverted_blurred_img = 255 - blurred_img
    pencil_sketch = cv2.divide(gray_image, inverted_blurred_img, scale=256.0)
    return pencil_sketch

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    
    sketch = convert_to_sketch(img)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    output_file = f'data:image/jpg;base64,{img_base64}'

    _, sketch_encoded = cv2.imencode('.jpg', sketch)
    sketch_base64 = base64.b64encode(sketch_encoded).decode('utf-8')
    sketch_file = f'data:image/jpg;base64,{sketch_base64}'
    
    return render_template('index.html', output=output_file, sketch=sketch_file)

if __name__ == "__main__":
    app.run(debug=True)
