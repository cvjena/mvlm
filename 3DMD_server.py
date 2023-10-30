import fastapi
import pydantic

import deepmvlm
import argparse
from parse_config import ConfigParser

try:
    from xvfbwrapper import Xvfb # for headless rendering
    vdisplay = Xvfb(width=256, height=256)
    vdisplay.start()
    print("Started virtual display with Xvfb")
except:
    print("Could not start virtual display with Xvfb")
    pass

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
# for debug, LIBGL_DEBUG=verbose before

# NOTE this might not work on other machines via ssh. to solve this:
#   - in the /anaconda3/envs/<name>/lib/libstdcc.so.6 rename (preferred) or delete the file
#   - and don't lock the screen on a machine with x-server running and monitor attached
#   - install xvfbwrapper (via pypi) and xvfb (via apt-get)