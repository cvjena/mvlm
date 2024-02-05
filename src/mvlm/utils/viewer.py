__all__ = ["VTKViewer"] 
import vtk
import numpy as np
import math

from .utils3d import obj_to_actor

class VTKViewer:
    def __init__(
        self,
        filename: str,
        landmarks: np.ndarray = None,
    ):
        # Initialize Camera
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)

        # Initialize RenderWindow
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetSize(1024, 1024)
        self.ren_win.SetOffScreenRendering(0)
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
        
        
        if landmarks is not None:
            lm_pd = self.get_landmarks_as_spheres(landmarks)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(lm_pd)

            actor_lm = vtk.vtkActor()
            actor_lm.SetMapper(mapper)
            actor_lm.GetProperty().SetColor(0, 0, 1)
            self.ren.AddActor(actor_lm)
            
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().SetPosition(0, 0, 1)
        self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.ren.GetActiveCamera().SetViewUp(0, 1, 0)
        # self.ren.GetActiveCamera().SetParallelProjection(1)
    
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.ren_win)
        self.iren.Initialize()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        
        self.ren_win.Render()
        self.iren.Start()
        
    def get_landmark_bounds(self, lms):
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
    
    def get_landmarks_bounding_box_diagonal_length(self, lms):
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_landmark_bounds(lms)

        diag_len = math.sqrt(
            (x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min) + (z_max - z_min) * (z_max - z_min))
        return diag_len
        
    def get_landmarks_as_spheres(self, lms):
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