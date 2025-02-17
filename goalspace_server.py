import asyncio
import logging

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
from ruamel.yaml import YAML

yaml = YAML(typ="safe")

from goals import DEFAULT_GOALS, Goalspace

GOALSPACE = Goalspace(DEFAULT_GOALS, cache_path="cache/rl_goalspace_cache_v2.json") 
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

def on_receive(endpoint, request_body):
    logging.info(f'Received request: {request_body} - "{endpoint}" HTTP/1.1')

app = FastAPI()

class InferenceRequest(BaseModel):
    texts: list[str]
    normalize: bool = True

@app.get("/health")
async def health():
    on_receive("/health", None)
    """Check server readiness."""
    return {"ready": True}

@app.get("/goals")
async def goals():
    on_receive("/goals", None)
    return {"goals": GOALSPACE.get_goal_names(snake_case=True)}

@app.post("/goalspace")
async def goalspace(request: InferenceRequest):
    on_receive("/goalspace", request.model_dump_json())
    try:
        results_df = GOALSPACE(request.texts)
        if request.normalize:
            for goal in results_df.columns:
                goal_min, goal_max = NORMALIZATION[goal]["min"], NORMALIZATION[goal]["max"]
                results_df[goal] = np.clip((results_df[goal] - goal_min) / (goal_max - goal_min), 0., 1.) # shape: (num_completions, num_goal_dims)
        results_json = results_df.to_dict(orient="list")
        return results_json
    except Exception as e:
        import traceback
        logging.error(f"500 Internal Server Error - {traceback.print_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")