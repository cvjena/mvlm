import argparse
import sys
from pathlib import Path
import numpy as np

import pandas as pd
import mvlm

parser = argparse.ArgumentParser()

# we adapt the already existing config parser and add our arguments
parser.add_argument('-c', '--config', default="configs/BU_3DFE-RGB+depth.json", type=str) 
parser.add_argument('-p', '--path', type=str, required=True)
parser.add_argument('-o', '--out', type=str, required=False)
args = parser.parse_args()

if args.out is None:
    args.out = args.path

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
# dm = mvlm.pipeline.MediaPipePipeline(render_image_stack=True, render_image_folder="visualization")
# dm = mvlm.pipeline.BU3DFEPipeline(render_image_stack=True, render_image_folder="visualization")
dm = mvlm.pipeline.DTU3DPipeline(render_image_stack=True, render_image_folder="visualization")
# dm = mvlm.pipeline.DlibPipeline(render_image_stack=True, render_image_folder="visualization")
# create the data frame for saving the landmarks in a csv file
columns_3D: list = [f"{i+1}.{l}" for i in range(dm.get_lm_count()) for l in ["x", "y", "z"]]

df = pd.DataFrame(index=columns_3D)

for i, file in enumerate(objFiles):
    print(f"Current file: {file}")

    # predict the landmarks
    landmarks = dm.predict_one_file(file)
    # insert them into the dataframe and save it
    df.insert(loc=i, column=i, value=landmarks.flatten())
    # df.transpose().to_csv((pathToOut / f"{file.stem}.csv").as_posix(), na_rep="nan", index_label="index")

    # np.save((pathToOut / f"{file.stem}.npy").as_posix(), landmarks)
    np.savetxt((pathToOut / f"{file.stem}.txt").as_posix(), landmarks, delimiter=",")

    # visualize the mesh
    mvlm.utils.VTKViewer(file.as_posix(), landmarks)
