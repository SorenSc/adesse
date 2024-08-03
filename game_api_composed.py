import json
import logging
import torch

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from game.game_api.src import game_simulator

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.info('GameToSimulatorConnector: Setting up the GameToSimulatorConnector...')
connector = game_simulator.GameToSimulatorConnector(True)


@app.get("/step/")
def step(run_id=0, action_x=0, action_y=0, scenario='A', idx=629, taxi_x=8, taxi_y=8):
    return json.dumps(connector.step(run_id, torch.tensor([int(action_x), int(action_y)]), scenario, int(idx), int(taxi_x), int(taxi_y)))


@app.post("/request_estimation_exp_selection/")
def dp_exp_selection(selection=9):
    print(f'Selected {selection}')
    connector.manage_request_estimation_exp_selection(selection)


if __name__ == "__main__":
    uvicorn.run("game_api:app", host="127.0.0.1", port=5000, log_level="info")
