__all__ = ["VTKViewer"]
import vtk
import numpy as np
import math
from pathlib import Path

from .utils3d import obj_to_actor


class VTKViewer:
    def __init__(
        self,
        filename: str,
        landmarks: np.ndarray | None = None,
        pname: str | None = None,
        save: bool = False,
    ) -> None:
        self.pname = pname
        self.filename = Path(filename)

        # Initialize Camera
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)

        # Initialize RenderWindow
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetSize(1024, 1024)
        self.ren_win.SetOffScreenRendering(1 if save else 0)
        self.ren_win.AddRenderer(self.ren)

        actor, pd = obj_to_actor(filename)
        self.ren.AddActor(actor)

        # center of mass
        center = vtk.vtkCenterOfMass()
        center.SetInputData(pd)
        center.SetUseScalarsAsWeights(False)
        center.Update()
        com = center.GetCenter()
        translation = [-com[0], -com[1], -com[2]]

        t = vtk.vtkTransform()
        t.Identity()

        rx = 0
        ry = 0
        rz = 0
        s = 1

        t.Scale(s, s, s)
        t.RotateY(ry)
        t.RotateX(rx)
        t.RotateZ(rz)
        t.Translate(translation)
        t.Update()

        # Transform (assuming only one mesh)
        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()

        self.landmarks_actor = None
        if landmarks is not None:
            lm_pd = self.get_landmarks_as_spheres(landmarks)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(lm_pd)

            self.landmarks_actor = vtk.vtkActor()
            self.landmarks_actor.SetMapper(mapper)
            self.landmarks_actor.GetProperty().SetColor(0, 0, 1)
            self.ren.AddActor(self.landmarks_actor)

        self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.ren.GetActiveCamera().SetViewUp(0, 1, 0)
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Zoom(1.4)
        self.ren.ResetCameraClippingRange()

        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)
        self.iren.Initialize()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        # Call setup_hotkeys during initialization
        self.setup_hotkeys()

        self.ren_win.Render()
        if save:
            stem = self.filename.stem
            suffix = f"_{self.pname}" if self.pname else ""
            out_path = Path("visualization") / f"{stem}{suffix}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self.take_screenshot(str(out_path))
        else:
            self.iren.Start()

    def get_landmark_bounds(self, lms: np.ndarray) -> tuple[float, float, float, float, float, float]:
        x_min = lms[0][0]
        x_max = x_min
        y_min = lms[0][1]
        y_max = y_min
        z_min = lms[0][2]
        z_max = z_min

        for lm in lms:
            x = lm[0]
            y = lm[1]
            z = lm[2]
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)
            z_min = min(z_min, z)
            z_max = max(z_max, z)

        return x_min, x_max, y_min, y_max, z_min, z_max

    def get_landmarks_bounding_box_diagonal_length(self, lms: np.ndarray) -> float:
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_landmark_bounds(lms)

        diag_len = math.sqrt((x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min) + (z_max - z_min) * (z_max - z_min))
        return diag_len

    def get_landmarks_as_spheres(self, lms: np.ndarray) -> vtk.vtkPolyData:
        diag_len = self.get_landmarks_bounding_box_diagonal_length(lms)
        # sphere radius is 0.8% of bounding box diagonal
        sphere_size = diag_len * 0.008

        append = vtk.vtkAppendPolyData()
        for idx in range(len(lms)):
            lm = lms[idx]
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(lm)
            sphere.SetRadius(sphere_size)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            sphere.Update()
            append.AddInputData(sphere.GetOutput())
            del sphere

        append.Update()
        return append.GetOutput()

    def take_screenshot(self, filename: str | None = None) -> None:
        if filename is None:
            filename = f"screenshot_{self.pname}.png"

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.ren_win)
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    def setup_hotkeys(self) -> None:
        def keypress_callback(obj: vtk.vtkRenderWindowInteractor, event: str) -> None:
            key = obj.GetKeySym()
            if key == "s":  # Press 's' to take a screenshot
                self.take_screenshot()
                print("Screenshot saved as 'screenshot.png'")
            elif key == "h":  # Press 'h' to toggle landmarks
                if self.landmarks_actor is not None:
                    is_visible = self.landmarks_actor.GetVisibility()
                    self.landmarks_actor.SetVisibility(not is_visible)
                    self.ren_win.Render()
                    print(f"Landmarks {'shown' if not is_visible else 'hidden'}")

        self.iren.AddObserver("KeyPressEvent", keypress_callback)  # type: ignore
