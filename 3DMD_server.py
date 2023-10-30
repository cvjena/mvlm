import fastapi
import pydantic

import deepmvlm
import argparse
from parse_config import ConfigParser


class Item(pydantic.BaseModel):
    path_in: str

class Landmarks(pydantic.BaseModel):
    landmarks: list[list[float]]

app = fastapi.FastAPI()

args = argparse.Namespace()
args.config = "configs/BU_3DFE-RGB+depth.json"

config = ConfigParser(args)

dm = deepmvlm.DeepMVLM(config)

@app.post("/landmarks3d")
def get_3d_landmarks(item: Item):
    landmarks = dm.predict_one_file(item.path_in)
    landmarks = landmarks.tolist()
    return Landmarks(landmarks=landmarks)

# start with
# uvicorn 3DMD_server:app --port 10000 --host 0.0.0.0   