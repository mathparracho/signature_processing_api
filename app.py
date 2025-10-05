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

app = Flask(__name__)
swagger = Swagger(app)

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
    Process two signature images and return comparison figures.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: signature1
        in: formData
        type: file
        required: true
        description: First signature image (PNG or JPG)
      - name: signature2
        in: formData
        type: file
        required: true
        description: Second signature image (PNG or JPG)
    responses:
      200:
        description: Returns 3 generated comparison images.
        schema:
          type: object
          properties:
            jaccard_image:
              type: string
              example: /compare/image/jaccard
            pressure_comparison:
              type: string
              example: /compare/image/pressure
            vector_comparison:
              type: string
              example: /compare/image/vector
    """
    if 'signature1' not in request.files or 'signature2' not in request.files:
        return jsonify({"error": "Both 'signature1' and 'signature2' files are required."}), 400

    try:
        # Salvar temporariamente as imagens enviadas
        os.makedirs("./uploads", exist_ok=True)
        img1_path = "./uploads/temp1.png"
        img2_path = "./uploads/temp2.png"

        Image.open(request.files['signature1']).save(img1_path)
        Image.open(request.files['signature2']).save(img2_path)

        # Chama função de processamento (gera as imagens)
        processing_pairs(img1_path, img2_path)

        # Caminhos das imagens geradas
        results = {
            "jaccard_image": "/compare/image/jaccard",
            "pressure_comparison": "/compare/image/pressure",
            "vector_comparison": "/compare/image/vector"
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    processing_pairs()
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)