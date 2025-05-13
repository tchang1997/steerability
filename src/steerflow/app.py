from flask import Flask, render_template, request, jsonify
from functools import lru_cache
from io import BytesIO
import json
import os
import pandas as pd

from steerability.utils.result_utils import STEERING_GOALS, print_steerability_summary
from steerflow.plotting_utils import grab_subspace, export_vector_field

HERE = os.path.abspath(os.path.dirname(__file__))  # steerflow/
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # src/
STATIC_DIR = os.path.join(HERE, "static")


USE_R2 = os.environ.get("STEERFLOW_USE_R2", "false").lower() == "true"

if USE_R2:
    import boto3
    import hashlib
    from botocore.client import Config

    ACCOUNT_ID = os.environ.get('R2_ACCOUNT_ID')
    ACCESS_KEY_ID = os.environ.get('R2_ACCESS_KEY_ID')
    SECRET_ACCESS_KEY = os.environ.get('R2_SECRET_ACCESS_KEY')
    BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')

    hashed_secret_key = hashlib.sha256(SECRET_ACCESS_KEY.encode()).hexdigest()
    s3_client = boto3.client('s3',
        endpoint_url=f'https://{ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=hashed_secret_key,
        config=Config(signature_version='s3v4')
    )
else:
    RESULTS_DIR = os.environ.get("STEERFLOW_RESULTS_DIR", os.path.join(ROOT, "results", "judged"))
    JSON_DIR = os.environ.get("STEERFLOW_JSON_DIR", os.path.join(ROOT, "results", "steerability_metrics"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def cached_list_objects():
    logger.info(f"Fetching object list from R2 for bucket: {BUCKET_NAME}")
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    return [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".csv")]

@lru_cache(maxsize=128)
def get_cached_object(filename: str):
    logger.info(f"Fetching r2://{filename}")
    return s3_client.get_object(Bucket=BUCKET_NAME, Key=filename)["Body"].read()

def get_df(filename):
    if USE_R2:
        logger.info(f"Reading result CSV: r2://{filename}")
        obj = get_cached_object(filename)
        df = pd.read_csv(BytesIO(obj), index_col=0)
    else:
        results_path = os.path.join(RESULTS_DIR, filename)
        logger.info(f"Reading result CSV from local: {results_path}")
        df = pd.read_csv(results_path, index_col=0)
    return df

def create_app():
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")

    @app.route("/")
    def index():
        if USE_R2:
            files = cached_list_objects()
            results_dir = f"r2://{BUCKET_NAME}"
        else:
            results_dir = RESULTS_DIR
            files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
        logger.info(f"Pulling results from {results_dir} (found {len(files)} results)")
        return render_template("index.html", files=files)

    @app.route("/columns", methods=["POST"])
    def get_columns():
        filename = request.json["filename"]
        df = get_df(filename)

        # Only return relevant columns (e.g., numeric)
        cols = [c[len("delta_"):] for c in df.columns if c.startswith("delta_")]
        logger.info(f"Found goal dimensions (delta_*): {cols}")
        return jsonify(cols)

    @app.route("/data", methods=["POST"])
    def get_data():
        payload = request.json
        filename = payload["filename"]
        df = get_df(filename) 

        x, y = payload["xcol"], payload["ycol"]
        x0 = df[f"source_{x}"].tolist()
        y0 = df[f"source_{y}"].tolist()
        x1 = df[f"target_{x}"].tolist()
        y1 = df[f"target_{y}"].tolist()
        text = df.get("tooltip", [""] * len(x0))
        return jsonify({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    
    @app.route("/summary", methods=["POST"])
    def get_steerability_printout():
        filename = request.json["filename"]
        json_path = os.path.join(JSON_DIR, filename.replace(".csv", ".json"))

        if not os.path.exists(json_path):
            return jsonify({"summary_html": "<pre style='color:red;'>Summary file not found.</pre>"}), 404

        with open(json_path) as f:
            steer_stats = json.load(f)

        summary_html = print_steerability_summary(steer_stats, stdout=False, return_html=True)
        return jsonify({"summary_html": summary_html})
    
    @app.route("/steerability_values", methods=["POST"])
    def steerability_values():
        filename = request.json["filename"]
        json_path = os.path.join(JSON_DIR, filename.replace(".csv", ".json"))

        if not os.path.exists(json_path):
            return jsonify({"error": "File not found"}), 404

        with open(json_path) as f:
            stats = json.load(f)

        steer_stats = stats["steerability"]
        return jsonify({
            "steering_error": steer_stats["steering_error"]["raw"],
            "miscalibration": steer_stats["miscalibration"]["raw"],
            "orthogonality": steer_stats["orthogonality"]["raw"],
        })
    
    @app.route("/generate_flow", methods=["POST"])
    def generate_flow():
        data = request.json
        filename = data["filename"]
        xcol, ycol = data["xcol"], data["ycol"]

        logger.info(f"Reading steerability probe results from {filename}")
        df = get_df(filename)

        logger.info(f"Grabbing subspace: ({xcol}, {ycol})")
        subspace = grab_subspace(df, xcol, ycol, steering_goals=STEERING_GOALS)

        output_path = os.path.join(STATIC_DIR, "_field.json")
        logger.info(f"Exporting vector field to {output_path}")
        export_vector_field(subspace, xcol, ycol, output_path=output_path)

        return jsonify({"status": "ok"})

    return app

