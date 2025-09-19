"""
FastAPI backend for Neural Style Transfer application.
Provides REST API endpoints for style transfer operations.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import os
import uuid
import time
from typing import List, Optional
import shutil
from pathlib import Path

from ..models.neural_style_transfer import StyleTransferPipeline
from ..database.models import DatabaseManager

app = FastAPI(
    title="Neural Style Transfer API",
    description="Modern neural style transfer with multiple algorithms",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
style_pipeline = StyleTransferPipeline()
db_manager = DatabaseManager()

# Create directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
ASSETS_DIR = Path("assets")

for directory in [UPLOAD_DIR, OUTPUT_DIR, ASSETS_DIR]:
    directory.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neural Style Transfer API",
        "version": "2.0.0",
        "endpoints": {
            "transfer": "/transfer",
            "templates": "/templates",
            "history": "/history",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload an image file"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate unique filename
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "filename": unique_filename,
        "path": str(file_path),
        "url": f"/uploads/{unique_filename}"
    }


@app.post("/transfer")
async def style_transfer(
    background_tasks: BackgroundTasks,
    content_file: UploadFile = File(...),
    style_file: UploadFile = File(...),
    method: str = "adain",
    alpha: float = 1.0,
    steps: int = 300
):
    """Perform style transfer"""
    
    # Validate inputs
    if method not in ["adain", "optimization"]:
        raise HTTPException(status_code=400, detail="Method must be 'adain' or 'optimization'")
    
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=400, detail="Alpha must be between 0 and 1")
    
    try:
        # Save uploaded files
        content_path = UPLOAD_DIR / f"content_{uuid.uuid4()}.jpg"
        style_path = UPLOAD_DIR / f"style_{uuid.uuid4()}.jpg"
        output_path = OUTPUT_DIR / f"output_{uuid.uuid4()}.jpg"
        
        with open(content_path, "wb") as f:
            shutil.copyfileobj(content_file.file, f)
        
        with open(style_path, "wb") as f:
            shutil.copyfileobj(style_file.file, f)
        
        # Perform style transfer
        start_time = time.time()
        
        if method == "adain":
            result_path = style_pipeline.transfer_style_adain(
                str(content_path), str(style_path), str(output_path), alpha
            )
        else:
            result_path = style_pipeline.transfer_style_optimization(
                str(content_path), str(style_path), str(output_path), steps
            )
        
        processing_time = time.time() - start_time
        
        # Save to database
        parameters = {"alpha": alpha, "steps": steps, "method": method}
        result_id = db_manager.save_transfer_result(
            str(content_path), str(style_path), str(output_path),
            method, parameters, processing_time
        )
        
        return {
            "result_id": result_id,
            "output_url": f"/outputs/{output_path.name}",
            "processing_time": processing_time,
            "method": method,
            "parameters": parameters
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")


@app.post("/transfer/template/{template_id}")
async def style_transfer_with_template(
    template_id: int,
    content_file: UploadFile = File(...),
    method: str = "adain",
    alpha: float = 1.0
):
    """Perform style transfer using a predefined template"""
    
    # Get template from database
    templates = db_manager.get_style_templates()
    template = next((t for t in templates if t['id'] == template_id), None)
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    try:
        # Save content file
        content_path = UPLOAD_DIR / f"content_{uuid.uuid4()}.jpg"
        output_path = OUTPUT_DIR / f"output_{uuid.uuid4()}.jpg"
        
        with open(content_path, "wb") as f:
            shutil.copyfileobj(content_file.file, f)
        
        # Perform style transfer
        start_time = time.time()
        
        if method == "adain":
            result_path = style_pipeline.transfer_style_adain(
                str(content_path), template['style_image_path'], str(output_path), alpha
            )
        else:
            result_path = style_pipeline.transfer_style_optimization(
                str(content_path), template['style_image_path'], str(output_path)
            )
        
        processing_time = time.time() - start_time
        
        # Update template popularity
        db_manager.update_template_popularity(template_id)
        
        # Save to database
        parameters = {"alpha": alpha, "method": method, "template_id": template_id}
        result_id = db_manager.save_transfer_result(
            str(content_path), template['style_image_path'], str(output_path),
            method, parameters, processing_time
        )
        
        return {
            "result_id": result_id,
            "output_url": f"/outputs/{output_path.name}",
            "processing_time": processing_time,
            "template": template,
            "method": method
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Style transfer failed: {str(e)}")


@app.get("/templates")
async def get_style_templates(category: Optional[str] = None):
    """Get available style templates"""
    templates = db_manager.get_style_templates(category)
    return {"templates": templates}


@app.get("/templates/categories")
async def get_template_categories():
    """Get available template categories"""
    templates = db_manager.get_style_templates()
    categories = list(set(t['category'] for t in templates if t['category']))
    return {"categories": categories}


@app.get("/history")
async def get_transfer_history(limit: int = 20, offset: int = 0):
    """Get style transfer history"""
    results = db_manager.get_transfer_results(limit, offset)
    return {"results": results, "limit": limit, "offset": offset}


@app.get("/history/{result_id}")
async def get_transfer_result(result_id: int):
    """Get specific transfer result"""
    result = db_manager.get_transfer_result_by_id(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


@app.delete("/history/{result_id}")
async def delete_transfer_result(result_id: int):
    """Delete a transfer result and associated files"""
    result = db_manager.get_transfer_result_by_id(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Delete files
    for path_key in ['content_image_path', 'style_image_path', 'output_image_path']:
        file_path = Path(result[path_key])
        if file_path.exists():
            file_path.unlink()
    
    return {"message": "Result deleted successfully"}


@app.get("/stats")
async def get_statistics():
    """Get application statistics"""
    results = db_manager.get_transfer_results(limit=1000)
    templates = db_manager.get_style_templates()
    
    total_transfers = len(results)
    methods_used = {}
    avg_processing_time = 0
    
    if results:
        for result in results:
            method = result['method']
            methods_used[method] = methods_used.get(method, 0) + 1
            avg_processing_time += result['processing_time'] or 0
        
        avg_processing_time /= len(results)
    
    popular_templates = sorted(templates, key=lambda x: x['popularity_score'], reverse=True)[:5]
    
    return {
        "total_transfers": total_transfers,
        "methods_used": methods_used,
        "average_processing_time": avg_processing_time,
        "total_templates": len(templates),
        "popular_templates": popular_templates
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
