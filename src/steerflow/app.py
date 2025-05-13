from flask import Flask, render_template, request, jsonify
import os
import pandas as pd

from steerability.utils.result_utils import STEERING_GOALS
from steerflow.plotting_utils import grab_subspace, export_vector_field

HERE = os.path.abspath(os.path.dirname(__file__))  # steerflow/
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # src/
STATIC_DIR = os.path.join(HERE, "static")


USE_S3 = os.environ.get("STEERFLOW_USE_S3", "false").lower() == "true"

if USE_S3:
    import boto3

    S3_ENDPOINT = os.environ["STEERFLOW_S3_ENDPOINT"]
    S3_BUCKET = os.environ["STEERFLOW_S3_BUCKET"]
    S3_PREFIX = os.environ.get("STEERFLOW_S3_PREFIX", "results/")
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT)
else:
    RESULTS_DIR = os.environ.get("STEERFLOW_RESULTS_DIR", os.path.join(ROOT, "results", "judged"))


import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")

    @app.route("/")
    def index():
        if USE_S3:
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
            files = [
                obj["Key"]
                for obj in response.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ]
            files = [key[len(S3_PREFIX):] for key in files]  # strip prefix for display
        else:
            files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
        logger.info(f"Pulling results from {RESULTS_DIR} (found {len(files)} results)")
        return render_template("index.html", files=files)

    @app.route("/columns", methods=["POST"])
    def get_columns():
        filename = request.json["filename"]
        results_path = os.path.join(RESULTS_DIR, filename)
        logger.info(f"Reading result CSV: {results_path}")
        df = pd.read_csv(results_path, index_col=0)
        # Only return relevant columns (e.g., numeric)
        cols = [c[len("delta_"):] for c in df.columns if c.startswith("delta_")]
        logger.info(f"Found goal dimensions (delta_*): {cols}")
        return jsonify(cols)

    @app.route("/data", methods=["POST"])
    def get_data():
        payload = request.json
        df = pd.read_csv(os.path.join(RESULTS_DIR, payload["filename"]))
        x, y = payload["xcol"], payload["ycol"]
        x0 = df[f"source_{x}"].tolist()
        y0 = df[f"source_{y}"].tolist()
        x1 = df[f"target_{x}"].tolist()
        y1 = df[f"target_{y}"].tolist()
        text = df.get("tooltip", [""] * len(x0))
        return jsonify({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    
    @app.route("/generate_flow", methods=["POST"])
    def generate_flow():
        data = request.json
        filename = data["filename"]
        xcol, ycol = data["xcol"], data["ycol"]

        logger.info(f"Reading steerability probe results from {filename}")
        df = pd.read_csv(os.path.join(RESULTS_DIR, filename))

        logger.info(f"Grabbing subspace: ({xcol}, {ycol})")
        subspace = grab_subspace(df, xcol, ycol, steering_goals=STEERING_GOALS)

        output_path = os.path.join(STATIC_DIR, "_field.json")
        logger.info(f"Exporting vector field to {output_path}")
        export_vector_field(subspace, xcol, ycol, output_path=output_path)

        return jsonify({"status": "ok"})

    return app

