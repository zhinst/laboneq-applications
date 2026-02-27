# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Flask server for automation graph web viewer."""

from __future__ import annotations

import logging
import threading
import webbrowser
from typing import TYPE_CHECKING

from laboneq_applications._automation.web_viewer.app.flask_app import (
    app,
    set_automation_instance,
)

if TYPE_CHECKING:
    from laboneq._automation import Automation

# Suppress Flask's default logging
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


def run_server(
    automation: Automation,
    port: int = 5000,
    host: str = "127.0.0.1",
) -> None:
    """Run the Flask server.

    This function is meant to be run in a background thread.

    Arguments:
        automation: The automation instance to visualize.
        port: The port to run the server on.
        host: The host to bind to.
    """
    set_automation_instance(automation)

    app.run(host=host, port=port, debug=True, use_reloader=False)


def start_web_viewer(
    automation: Automation,
    port: int = 5000,
    host: str = "127.0.0.1",
    *,
    open_browser: bool = False,
) -> None:
    """Start an interactive web viewer for the automation graph.

    The web viewer provides an interactive D3.js visualization of the automation
    graph with zoom, pan, and clickable nodes. The graph automatically refreshes
    when changes are detected.

    Arguments:
        automation: The workflow automation instance.
        port: The port to run the web server on (default: 5000).
        host: The host to bind to (default: 127.0.0.1).
        open_browser: Whether to automatically open the browser (default: False).

    Raises:
        RuntimeError: If the web viewer is already running.
    """
    thread = threading.Thread(
        target=run_server,
        args=(automation, port, host),
        daemon=True,
    )
    thread.start()

    url = f"http://{host}:{port}"
    print(f"Web viewer started at {url}")  # noqa: T201

    if open_browser:
        # Give the server a moment to start
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
