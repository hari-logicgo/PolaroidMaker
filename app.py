import os
import io
import traceback
from datetime import datetime
from typing import Optional

from fastapi import (
    FastAPI, File, UploadFile, Form,
    HTTPException, Security
)
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from PIL import Image
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from dotenv import load_dotenv
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ---------------------------------------------------------------------
# Load environment variables
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
# ---------------------------------------------------------------------
hf_client = InferenceClient(token=HF_TOKEN)

# ---------------------------------------------------------------------
# Initialize MongoDB connection + GridFS
# ---------------------------------------------------------------------
mongo = MongoClient(MONGODB_URI)
db = mongo[DB_NAME]
fs = gridfs.GridFS(db)
logs_collection = db["logs"]

# ---------------------------------------------------------------------
# FastAPI setup
# ---------------------------------------------------------------------
app = FastAPI(
    title="Polaroid AI Generator",
    description="Upload an image + prompt to generate a styled version. "
                "Use the **Authorize 🔒** button to provide your API token before generating.",
    version="1.0.0"
)

# Enable CORS (for local & HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Authentication setup (with Swagger integration)
# ---------------------------------------------------------------------
BEARER_TOKEN = "logicgo@123"
security_scheme = HTTPBearer(auto_error=False)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    """Bearer token verification (used in Swagger UI too)."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    token = credentials.credentials
    if token != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid bearer token.")
    return True

# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    hf_provider: str
    db: str

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    try:
        mongo.admin.command("ping")
        return HealthResponse(status="ok", hf_provider="huggingface", db=db.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e}")


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    authorized: bool = Security(verify_token)
):
    """
    Upload an image and prompt -> get AI-edited image.
    Requires Bearer token auth.
    """
    # 1️⃣ Read input file
    try:
        input_bytes = await file.read()
        if not input_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading upload: {e}")

    # 2️⃣ Save input to GridFS
    try:
        input_meta = {"filename": file.filename, "contentType": file.content_type, "role": "input"}
        input_id = fs.put(input_bytes, **input_meta)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed saving input image: {e}")

    # 3️⃣ Run inference
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

    # 4️⃣ Save output image
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

    # 5️⃣ Log operation
    try:
        logs_collection.insert_one({
            "timestamp": datetime.utcnow(),
            "input_image_id": str(input_id),
            "output_image_id": str(out_id),
            "prompt": prompt,
        })
    except Exception as e:
        print("⚠️ Failed to write log:", e)

    # 6️⃣ Return result
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
    return {
        "message": "Polaroid Image Generator API running ✅",
        "routes": ["/health", "/generate", "/image/{id}"]
    }

# ---------------------------------------------------------------------
# Run the FastAPI app locally
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
