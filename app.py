from flask import Flask, request, jsonify
import torch
from architecture import ContrastiveNetwork
from PIL import Image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from flasgger import Swagger
from processing.main import processing_pairs
import base64
import io
import os

app = Flask(__name__)
swagger = Swagger(app)  ## http://194.163.172.45/apidocs/ || http://localhost/apidocs/

model = ContrastiveNetwork()
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    #A.Lambda(
    #    image=binarize_image
    #),
    ToTensorV2(),
])

def encode_image_base64(image_path: str):
    """Converte uma imagem em base64 para envio no JSON"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def resize_keep_aspect(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    w, h = img.size

    if max(w, h) <= max_dim:
        return img

    scale = max_dim / float(max(w, h))
    new_w = int(w * scale)
    new_h = int(h * scale)

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

@app.route("/")
def index():
    """
    Basic health check route to confirm the API is running.
    ---
    responses:
      200:
        description: Returns a success message
        schema:
          type: string
          example: API running!
    """
    return "API running!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Compare two signatures and return the Euclidean distance between their embeddings.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: signature1
        in: formData
        type: file
        required: true
        description: First signature image (PNG or JPG).
      - name: signature2
        in: formData
        type: file
        required: true
        description: Second signature image (PNG or JPG).
    responses:
      200:
        description: Euclidean distance between the embeddings.
        schema:
          type: object
          properties:
            distance:
              type: number
              example: 0.134
      400:
        description: Bad request (missing or invalid inputs).
      500:
        description: Internal server error.
    """
    if 'signature1' not in request.files or 'signature2' not in request.files:
        return jsonify({"error": "Both 'signature1' and 'signature2' files are required."}), 400

    try:
        img1 = np.asarray(Image.open(request.files['signature1']).convert("L"))
        img2 = np.asarray(Image.open(request.files['signature2']).convert("L"))

        tensor1 = transform(image=img1)["image"].unsqueeze(0)
        tensor2 = transform(image=img2)["image"].unsqueeze(0)

        with torch.no_grad():
            emb1, emb2 = model(tensor1, tensor2)
            distance = F.pairwise_distance(emb1, emb2).item()

        return jsonify({"distance": distance})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    """
    Process two signature images and return the generated comparison figures encoded in Base64.

    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: signature1
        in: formData
        type: file
        required: true
        description: First signature image (PNG or JPG).
      - name: signature2
        in: formData
        type: file
        required: true
        description: Second signature image (PNG or JPG).
    responses:
      200:
        description: Returns the three generated comparison images encoded in Base64.
        schema:
          type: object
          properties:
            jaccard_image:
              type: string
              description: Base64-encoded image showing the Jaccard index comparison between the two signatures.
              example: "iVBORw0KGgoAAAANSUhEUgAA..."
            pressure_comparison:
              type: string
              description: Base64-encoded image showing the comparison of pressure distributions.
              example: "iVBORw0KGgoAAAANSUhEUgAA..."
            vector_comparison:
              type: string
              description: Base64-encoded image showing vector direction and flow comparison.
              example: "iVBORw0KGgoAAAANSUhEUgAA..."
      400:
        description: Bad request (missing or invalid inputs).
      500:
        description: Internal server error.
    """

    if 'signature1' not in request.files or 'signature2' not in request.files:
        return jsonify({"error": "Both 'signature1' and 'signature2' files are required."}), 400

    try:
        os.makedirs("./uploads", exist_ok=True)
        img1_path = "./uploads/temp1.png"
        img2_path = "./uploads/temp2.png"

        img1 = Image.open(request.files['signature1']).convert("L")
        img2 = Image.open(request.files['signature2']).convert("L")

        print("img1 size antes:", img1.size)
        print("img2 size antes:", img2.size)

        img1 = resize_keep_aspect(img1, max_dim=1024)
        img2 = resize_keep_aspect(img2, max_dim=1024)

        print("img1 size depois:", img1.size)
        print("img2 size depois:", img2.size)

        img1.save(img1_path)
        img2.save(img2_path)

        processing_pairs(img1_path, img2_path)

        # Caminhos esperados de saída
        jaccard_path = "./jaccard.png"
        pressure_path = "./pressure_comparison.png"
        vector_path = "./vector_comparison.png"

        # Converter para base64
        jaccard_b64 = encode_image_base64(jaccard_path)
        pressure_b64 = encode_image_base64(pressure_path)
        vector_b64 = encode_image_base64(vector_path)

        # Retornar JSON
        return jsonify({
            "jaccard_image": jaccard_b64,
            "pressure_comparison": pressure_b64,
            "vector_comparison": vector_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)