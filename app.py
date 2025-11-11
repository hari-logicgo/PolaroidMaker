import os
import io
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from PIL import Image
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load .env (for local development)
# ---------------------------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "polaroid")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set. Please add it in your .env file.")
if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI not set. Please add it in your .env file.")

# ---------------------------------------------------------------------
# Initialize Hugging Face Inference client
# (provider parameter only supported on Hugging Face Spaces)
# For local, just use the standard client
# ---------------------------------------------------------------------
hf_client = InferenceClient(token=HF_TOKEN)

# ---------------------------------------------------------------------
# Initialize MongoDB connection + GridFS
# ---------------------------------------------------------------------
mongo = MongoClient(MONGODB_URI)
db = mongo[DB_NAME]  # explicitly select DB
fs = gridfs.GridFS(db)

# ---------------------------------------------------------------------
# Create FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="Flux/FAL Image Inference Service")

# Enable CORS (for local testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Models and Endpoints
# ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    hf_provider: str
    db: str


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    try:
        mongo.admin.command("ping")
        return HealthResponse(status="ok", hf_provider="huggingface", db=db.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e}")


@app.post("/generate")
async def generate(prompt: str = Form(...), file: UploadFile = File(...)):
    """
    Upload an image and a prompt, get back an output image stored in MongoDB GridFS.
    Returns the output image's ID.
    """
    try:
        input_bytes = await file.read()
        if not input_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading upload: {e}")

    # Save input image to DB
    try:
        input_meta = {"filename": file.filename, "contentType": file.content_type, "role": "input"}
        input_id = fs.put(input_bytes, **input_meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed saving input image: {e}")

    # Run inference (image editing)
    try:
        pil_result = hf_client.image_to_image(
            image=input_bytes,
            prompt=prompt,
            model="Qwen/Qwen-Image-Edit"
        )

        if isinstance(pil_result, list):
            pil = pil_result[0]
        else:
            pil = pil_result

        out_buf = io.BytesIO()
        pil.save(out_buf, format="PNG")
        out_bytes = out_buf.getvalue()

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # Save output image
    try:
        out_meta = {
            "filename": f"result_{input_id}.png",
            "contentType": "image/png",
            "prompt": prompt,
            "input_id": str(input_id),
            "role": "output",
        }
        out_id = fs.put(out_bytes, **out_meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed saving output image: {e}")

    return JSONResponse({"output_id": str(out_id)})


@app.get("/image/{image_id}")
def get_image(image_id: str, download: Optional[bool] = False):
    """Retrieve image bytes from GridFS."""
    try:
        oid = ObjectId(image_id)
        grid_out = fs.get(oid)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

    def iterfile():
        yield grid_out.read()

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{grid_out.filename}"'

    return StreamingResponse(
        iterfile(),
        media_type=grid_out.content_type or "application/octet-stream",
        headers=headers,
    )


@app.get("/")
def root():
    """Simple root route for status."""
    return {"message": "Image inference service running", "routes": ["/health", "/generate", "/image/{id}"]}


# ---------------------------------------------------------------------
# Run the FastAPI app locally (no Docker needed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
