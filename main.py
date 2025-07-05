import os
import json
import hashlib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from utils import create_agent, parse_preferences

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Directory Structure Generator API")

# Pydantic input model
class GenerateRequest(BaseModel):
    project_desc: str
    tech_stack: List[str]
    preferences: Optional[str] = ""

# Pydantic response model
class GenerateResponse(BaseModel):
    json_structure: dict
    tree_view: str

# Initialize agent once on startup
try:
    agent = create_agent()
except Exception as e:
    logger.error(f"Agent initialization failed: {e}")
    agent = None

@app.post("/generate", response_model=GenerateResponse)
def generate_directory_structure(data: GenerateRequest):
    if not agent:
        raise HTTPException(status_code=500, detail="DirectoryStructureAgent not initialized")

    try:
        tech_list = data.tech_stack
        prefs = parse_preferences(data.preferences or "")
        project_id = f"api_{hashlib.md5(data.project_desc.encode()).hexdigest()[:8]}"
        structure = agent.suggest_structure(project_id, data.project_desc, tech_list, prefs)

        if not structure:
            raise HTTPException(status_code=500, detail="Failed to generate directory structure")

        return GenerateResponse(
            json_structure=structure,
            tree_view=agent.structure_to_tree(structure)
        )
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")
