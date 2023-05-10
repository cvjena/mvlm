import argparse
import sys
from pathlib import Path

import pandas as pd

import deepmvlm
from parse_config import ConfigParser

parser = argparse.ArgumentParser()

# we adapt the already existing config parser and add our arguments
parser.add_argument('-c', '--config', default="configs/BU_3DFE-RGB.json", type=str) 
parser.add_argument('-p', '--path', type=str, required=True)
parser.add_argument('-o', '--out', type=str, required=True)
config = ConfigParser(parser)

args = parser.parse_args()

# handle all pathing tasks
pathToDir = Path(args.path) # should be the path containing the meshes
pathToOut = Path(args.out)  # most likely the upper folder of the meshes

if not pathToDir.is_dir():
    print(f"{pathToDir.as_posix()} is not a valid path.")
    sys.exit(1)

# create the output folder if does not exits
pathToOut.mkdir(parents=True, exist_ok=True)

# try to find all .obj files in the given folder
objFiles = sorted(pathToDir.glob("*.obj"))

if len(objFiles) == 0:
    print("Given path does not contain any obj files")
    sys.exit(1)

# create the model for predicting the landmarks
dm = deepmvlm.DeepMVLM(config)

# create the data frame for saving the landmarks in a csv file
columns_3D: list = [f"{i+1}.{l}" for i in range(84) for l in ["x", "y", "z"]]
df = pd.DataFrame(index=columns_3D)

for i, file in enumerate(objFiles):
    print(f"Current file: {file}")

    # predict the landmarks
    landmarks = dm.predict_one_file(file.as_posix())
    # insert them into the dataframe and save it
    df.insert(loc=i, column=i, value=landmarks.flatten())
    df.transpose().to_csv((pathToOut / "landmarks3D.csv").as_posix(), na_rep="nan", index_label="index")