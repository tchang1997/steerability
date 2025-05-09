from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import plotly.graph_objects as go

from steerability.utils.result_utils import STEERING_GOALS
from steerflow.plotting_utils import grab_subspace, export_vector_field

cwd = os.path.dirname(__file__)
RESULTS_DIR = os.path.abspath(os.path.join(cwd, "..", "..", "results", "judged"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)

    @app.route("/")
    def index():
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
        df = pd.read_csv(f"results/{payload['filename']}")
        x, y = payload["xcol"], payload["ycol"]
        x0 = df[f"source_{x}"].tolist()
        y0 = df[f"source_{y}"].tolist()
        x1 = df[f"target_{x}"].tolist()
        y1 = df[f"target_{y}"].tolist()
        text = df.get("tooltip", [""] * len(x0))
        return jsonify({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})

    @app.route("/plot", methods=["POST"])
    def plot():
        # Dummy data: 3 arrows starting at (x, y), pointing (u, v)
        x = [0, 0.1, 0.2]
        y = [0, 0.1, 0.2]
        u = [1, 0.5, 0.4]
        v = [0.5, 0.3, 1]

        # Arrow shapes for Plotly
        shapes = []
        data = request.json
        xcol = data["xcol"]
        ycol = data["ycol"]

        scatter = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=8, color="rgba(255, 100, 100, 0.7)"),
            hoverinfo="text",
            name="start"
        )

        shapes = []
        for i in range(len(x)):
            shapes.append(dict(
                type="line",
                x0=x[i],
                y0=y[i],
                x1=x[i] + u[i],
                y1=y[i] + v[i],
                line=dict(color="blue", width=2)
            ))

        layout = go.Layout(
            xaxis=dict(
                title=xcol,
                range=[0, 1],
                linecolor="black",
                ticks="outside",
                constrain="range",
                domain=[0, 1],
                showgrid=True,
                gridcolor="#e0e0e0",
                gridwidth=0.2,
            ),
            yaxis=dict(
                title=ycol,
                range=[0, 1],
                linecolor="black",
                ticks="outside",
                constrain="range",
                domain=[0, 1],
                scaleanchor="x",
                scaleratio=1,
                showgrid=True,
                gridcolor="#e0e0e0",
                gridwidth=0.2,
            ),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(t=10, r=10, b=40, l=40),
            showlegend=False,
            shapes=shapes  # if you're drawing arrows
        )

        fig = go.Figure(data=[scatter], layout=layout)
        return jsonify(fig.to_plotly_json())
    
    @app.route("/generate_flow", methods=["POST"])
    def generate_flow():
        data = request.json
        filename = data["filename"]
        xcol, ycol = data["xcol"], data["ycol"]

        logger.info(f"Reading steerability probe results from {filename}")
        df = pd.read_csv(os.path.join(RESULTS_DIR, filename))

        logger.info(f"Grabbing subspace: ({xcol}, {ycol})")
        subspace = grab_subspace(df, xcol, ycol, steering_goals=STEERING_GOALS)

        logger.info(f"Exporting vector field!")
        export_vector_field(subspace, xcol, ycol, output_path="static/_field.json")

        return jsonify({"status": "ok"})

    return app

