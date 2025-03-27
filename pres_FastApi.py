from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import os
import shutil
import json
from datetime import datetime
import uvicorn

# Import the existing prescription processing logic
from prescription import process_prescription, PrescriptionInformations
# Create FastAPI app
app = FastAPI(
    title="Medical Prescription Parser API",
    description="An API to parse medical prescription images",
    version="1.0.0",
    docs_url="/docs",  # Explicitly set Swagger UI path
    redoc_url="/redoc"  # Explicitly set ReDoc path
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/prescription/", response_model=dict)
async def prescription(file: UploadFile = File(...)):
    """
    Endpoint to parse a prescription image
    
    - Accepts a single image file upload
    - Processes the prescription 
    - Returns structured prescription information
    """
    # Create a unique temporary folder for this upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_folder = f"temp_uploads_{timestamp}"
    os.makedirs(temp_folder, exist_ok=True)
    
    try:
        # Save the uploaded file
        file_path = os.path.join(temp_folder, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the prescription
        result = process_prescription([file_path])
        
        # Clean up temporary files
        shutil.rmtree(temp_folder)
        
        return result
    
    except Exception as e:
        # Clean up temporary files in case of error
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        # Raise HTTP exception with details
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Simple health check endpoint"""
    return {"message": "Prescription Parser API is running"}

# Custom OpenAPI schema (optional, but can help)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Medical Prescription Parser API",
        version="1.0.0",
        description="An API to parse medical prescription images",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Main block to run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)