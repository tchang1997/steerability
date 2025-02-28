import asyncio
import logging
import uuid

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

from goals import DEFAULT_GOALS, Goalspace

GOALSPACE = Goalspace(DEFAULT_GOALS, cache_path="cache/rl_goalspace_cache_v5.json") 
with open("config/seed_data/magic_numbers.yml", "r") as f:
    NORMALIZATION = yaml.load(f)

log_file = "goalspace_server.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w")  
    ]
)

def on_receive(endpoint, request_id, request_body):
    logging.info(f'Received request {request_id}: {request_body} - "{endpoint}" HTTP/1.1')

def on_return(endpoint, request_id, response):
    logging.info(f'Sending response {request_id}: {response} - "{endpoint}" HTTP/1.1')

app = FastAPI()

class InferenceRequest(BaseModel):
    texts: list[str]
    normalize: bool = True

@app.get("/health")
async def health():
    on_receive("/health", None, None)
    """Check server readiness."""
    return {"ready": True}

@app.get("/goals")
async def goals():
    on_receive("/goals", None, None)
    return {"goals": GOALSPACE.get_goal_names(snake_case=True)}

@app.post("/goalspace")
async def goalspace(request: InferenceRequest):
    request_id = str(uuid.uuid4())
    on_receive("/goalspace", request_id, request.model_dump_json())
    try:
        results_df = GOALSPACE(request.texts)
        if request.normalize:
            for goal in results_df.columns:
                goal_min, goal_max = NORMALIZATION[goal]["min"], NORMALIZATION[goal]["max"]
                results_df[goal] = np.clip((results_df[goal] - goal_min) / (goal_max - goal_min), 0., 1.) # shape: (num_completions, num_goal_dims)
        results_json = results_df.to_dict(orient="list")
        on_return("/goalspace", request_id, results_json)
        return results_json
    except Exception as e:
        import traceback
        logging.error(f"500 Internal Server Error - {traceback.print_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
