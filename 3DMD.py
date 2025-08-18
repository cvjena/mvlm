import argparse
import sys
from pathlib import Path

import numpy as np

import mvlm

parser = argparse.ArgumentParser()

# we adapt the already existing config parser and add our arguments
parser.add_argument("-p", "--path", type=str, required=True)
parser.add_argument("-o", "--out", type=str, required=False)
parser.add_argument("-n", "--n-views", type=int, default=8, help="Number of views to render")
parser.add_argument("-vis", "--visualize", action="store_true", help="Visualize the meshes with the predicted landmarks")
parser.add_argument("--visualize-method", action="store_true", help="Visualize the meshes with the predicted landmarks using a specific method")

args = parser.parse_args()

if args.out is None:
    args.out = args.path

# handle all pathing tasks
pathToDir = Path(args.path)  # should be the path containing the meshes
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
    # for pname in ["mediapipe", "bu3dfe", "dlib", "dtu3d", "face_alignment"]:
    for pname in ["mediapipe"]:
        print(f"Pipeline: {pname}")
        print(f"Current file: {file}")

        dm = mvlm.pipeline.create_pipeline(pname, render_image_stack=args.visualize_method, n_views=args.n_views)
        # predict the landmarks
        landmarks = dm.predict_one_file(file)

        if landmarks is None:
            print(f"Landmarks for {file} could not be predicted -> skipping file [{file.stem}] for pipeline {pname}")
            continue

        np.savetxt((pathToOut / f"{file.stem}_{pname}.txt").as_posix(), landmarks, delimiter=",")

        if not args.visualize:
            continue
        # visualize the mesh
        mvlm.utils.VTKViewer(file.as_posix(), landmarks)
