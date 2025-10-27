"""server.py
Server to launch a FastAPI / Swagger UI instance.
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.config import SCANNER_DEFAULTS
from src.models import ResumeData
from src.parse_classes.resume_parse_framework import ResumeParserFramework


app = FastAPI(title="Simple Resume Parser API", version="1.0")

class ParseResumeInputs(BaseModel):
    file: str

# Initiate ResumeParserFramework for use when server calls
resume_parse_framework = ResumeParserFramework()

@app.post(
    "/parse_resume",
    response_model=ResumeData,
    summary="Parse a resume file and extract structured data",
    description="Uploads a resume (PDF or DOCX), processes it, and returns structured ResumeData.",
)
async def parse_resume(file: UploadFile = File(...)) -> ResumeData:
    """
    Upload a resume file, validate it, parse it, and return extracted ResumeData.
    """
    # ---- Validate file size ----
    contents = await file.read()
    max_bytes = SCANNER_DEFAULTS.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {SCANNER_DEFAULTS.MAX_FILE_SIZE_MB} MB.",
        )

    # ---- Create temp directory and save uploaded file ----
    os.makedirs("temp_files", exist_ok=True)
    temp_path = os.path.join("temp_files", file.filename)

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        # ---- Run parsing pipeline ----
        resume_data = resume_parse_framework.parse_resume(file_path=temp_path)

        # ---- Validate return type ----
        if not isinstance(resume_data, ResumeData):
            raise HTTPException(
                status_code=500,
                detail="Parsing failed: invalid ResumeData returned from parser.",
            )

        return resume_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # ---- Cleanup temp file ----
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass