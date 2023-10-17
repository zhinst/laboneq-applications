import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from tuneup.configs import get_config
from tuneup.helper import find_latest_file_with_uid

app = FastAPI()
config = get_config()


@app.get("/")
async def root():
    return {"message": "Qubit monitor service"}


parent_dir = Path.cwd().parent
QUBIT_DIR = parent_dir / Path(config.get("Settings", "qubit_save_dir"))


@app.get("/calibration/{file_name}")
async def get_calibration(file_name: str):
    file_path = QUBIT_DIR / file_name
    try:
        with open(file_path, "r") as file:
            calibration_data = json.load(file)
            return calibration_data
    except FileNotFoundError:
        return Response(
            status_code=204, content=f"Calibration file '{file_name}' not found"
        )


@app.get("/calibration/{qb_uid}/latest")
async def get_latest_calibration(qb_uid: str):
    try:
        latest_calib = find_latest_file_with_uid(QUBIT_DIR, qb_uid)
        if latest_calib is None:
            raise FileNotFoundError
        else:
            with open(latest_calib, "r") as file:
                calibration_data = json.load(file)
                return calibration_data
    except FileNotFoundError:
        return Response(
            status_code=204, content=f"Calibration file for qubit '{qb_uid}' not found"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
