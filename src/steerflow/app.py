from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import plotly.graph_objects as go

from steerability.utils.result_utils import STEERING_GOALS
from steerflow.plotting_utils import grab_subspace, export_vector_field

HERE = os.path.abspath(os.path.dirname(__file__))  # steerflow/
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # src/
RESULTS_DIR = os.path.join(ROOT, "results", "judged")
STATIC_DIR = os.path.join(HERE, "static")

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
        results_dir = RESULTS_DIR if "STEERFLOW_RESULTS_DIR" not in os.environ else os.environ["STEERFLOW_RESULTS_DIR"]
        files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]
        logger.info(f"Pulling results from {results_dir} (found {len(files)} results)")
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

