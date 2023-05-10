import fastapi
import pandas as pd
import pydantic

import deepmvlm
import argparse
from parse_config import ConfigParser


class Item(pydantic.BaseModel):
    path_in: str
    path_out: str

app = fastapi.FastAPI()

args = argparse.Namespace()
args.config = "configs/BU_3DFE-RGB+depth.json"

config = ConfigParser(args)

dm = deepmvlm.DeepMVLM(config)

@app.post("/landmarks3d")
def get_3d_landmarks(item: Item):
    path_in = item.path_in
    path_out = item.path_out
    landmarks = dm.predict_one_file(path_in)
    pd.DataFrame(landmarks, columns=["x", "y", "z"]).to_csv(path_out, index=False, header=True)
    return {"status": "ok"}
