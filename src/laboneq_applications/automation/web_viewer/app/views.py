# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Flask routes to the pages of the web viewer."""

from flask import Response, jsonify, render_template

from laboneq_applications.automation.web_viewer.app.flask_app import (
    app,
    get_automation_instance,
)
from laboneq_applications.automation.web_viewer.app.utils import export_graph_to_json


@app.route("/graph")
def get_graph() -> tuple[Response, int] | Response:
    """Return the automation graph as JSON."""
    automation = app.config.get("AUTOMATION_INSTANCE")
    if automation is None:
        return jsonify({"error": "No automation instance available"}), 500

    graph_data = export_graph_to_json(automation)
    return jsonify(graph_data)


@app.route("/")
def index() -> str:
    """Serve the main HTML page."""
    automation = app.config.get("AUTOMATION_INSTANCE")
    name = "Automation Graph" if automation is None else automation.name
    return render_template("index.html", name=name)


@app.route("/reset", methods=["POST"])
def reset_automation() -> tuple[Response, int]:
    """Reset the automation in a background thread."""
    automation = get_automation_instance()

    if automation is None:
        return jsonify({"error": "No automation instance available"}), 500

    automation.reset()
    return jsonify({"status": "reset"}), 202


@app.route("/run", methods=["POST"])
def run_automation() -> tuple[Response, int]:
    """Run the automation in a background thread."""
    automation = app.config.get("AUTOMATION_INSTANCE")

    if automation is None:
        return jsonify({"error": "No automation instance available"}), 500

    automation.run()
    return jsonify({"status": "started"}), 202
