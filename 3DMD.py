import argparse
import sys
from pathlib import Path
import numpy as np

import mvlm

parser = argparse.ArgumentParser()

# we adapt the already existing config parser and add our arguments
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

for i, file in enumerate(objFiles):
    for pname in ["mediapipe", "bu3dfe", "dlib", "dtu3d", "face_alignment"]:
        print(f"Pipeline: {pname}")
        print(f"Current file: {file}")
        
        dm = mvlm.pipeline.create_pipeline(pname, render_image_stack=True, render_image_folder="visualization")
        # predict the landmarks
        landmarks = dm.predict_one_file(file)
        np.savetxt((pathToOut / f"{file.stem}_{pname}.txt").as_posix(), landmarks, delimiter=",")

        # visualize the mesh
        mvlm.utils.VTKViewer(file.as_posix(), landmarks)