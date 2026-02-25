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
parser.add_argument("-vis-iter", "--visualize-iter", action="store_true", help="Interactively visualize each mesh with the predicted landmarks")
parser.add_argument("-vis-img", "--visualize-img", action="store_true", help="Render each mesh with predicted landmarks offscreen and save to visualization/")
parser.add_argument("--visualize-method", action="store_true", help="Visualize the meshes with the predicted landmarks using a specific method")

args = parser.parse_args()

if args.out is None:
    args.out = args.path

# handle all pathing tasks
inputPath = Path(args.path)
pathToOut = Path(args.out)

if not inputPath.exists():
    print(f"{inputPath.as_posix()} does not exist.")
    sys.exit(1)

# collect obj files from file or folder
if inputPath.is_file():
    if inputPath.suffix.lower() != ".obj":
        print(f"{inputPath.as_posix()} is not an .obj file.")
        sys.exit(1)
    objFiles = [inputPath]
    if args.out == args.path:
        pathToOut = inputPath.parent
else:
    pathToOut = Path(args.out)
    objFiles = sorted(inputPath.glob("*.obj"))
    if len(objFiles) == 0:
        print("Given folder does not contain any .obj files.")
        sys.exit(1)

# create the output folder if it does not exist
pathToOut.mkdir(parents=True, exist_ok=True)

for pname in ["mediapipe", "bu3dfe", "dlib", "dtu3d", "face_alignment"]:
    print(f"Pipeline: {pname}")
    dm = mvlm.pipeline.create_pipeline(pname, render_image_stack=args.visualize_method, n_views=args.n_views)
    for i, file in enumerate(objFiles):
        print(f"Current file: {file}")
        # predict the landmarks
        landmarks = dm.predict_one_file(file)

        if landmarks is None:
            print(f"Landmarks for {file} could not be predicted -> skipping file [{file.stem}] for pipeline {pname}")
            continue

        np.savetxt((pathToOut / f"{file.stem}_{pname}.txt").as_posix(), landmarks, delimiter=",")

        if args.visualize_iter:
            mvlm.utils.VTKViewer(file.as_posix(), landmarks, pname=pname, save=False)
        if args.visualize_img:
            mvlm.utils.VTKViewer(file.as_posix(), landmarks, pname=pname, save=True)
