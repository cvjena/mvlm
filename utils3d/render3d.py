__all__ = ["ObjVTKRenderer3D"]
import time
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from utils3d import obj_to_actor

class ObjVTKRenderer3D:
    def __init__(
        self,
        n_views: int = 9,
        image_size: tuple = (256, 256), 
        offscreen: bool = True,
        min_x_angle: int = -40,
        max_x_angle: int =  40,
        min_y_angle: int = -80,
        max_y_angle: int =  80,
        min_z_angle: int = -20,
        max_z_angle: int =  20,
        min_scale: float = 1.4,
        max_scale: float = 1.9,
        min_tx: int = -20,
        max_tx: int =  20,
        min_ty: int = -20,
        max_ty: int =  20,
    ):
        self.n_views = n_views
        self.image_size = image_size
        self.offscreen = offscreen
        self.min_x_angle = min_x_angle
        self.max_x_angle = max_x_angle
        self.min_y_angle = min_y_angle
        self.max_y_angle = max_y_angle
        self.min_z_angle = min_z_angle
        self.max_z_angle = max_z_angle
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_tx = min_tx
        self.max_tx = max_tx
        self.min_ty = min_ty
        self.max_ty = max_ty
        
        # fixed parameters 
        self.slack = 5
        # this still useless but more simplified from the original code
        self.side_length =  max([150 - (-150), 150 - (-150)]) * 1.0 / 2

        # Initialize Camera
        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(1, 1, 1)
        self.ren.GetActiveCamera().SetPosition(0, 0, 1)
        self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
        self.ren.GetActiveCamera().SetViewUp(0, 1, 0)
        self.ren.GetActiveCamera().SetParallelProjection(1)

        # Initialize RenderWindow
        self.ren_win = vtk.vtkRenderWindow()
        self.ren_win.SetSize(self.image_size[0], self.image_size[1])
        self.ren_win.SetShowWindow(0)
        self.ren_win.SetOffScreenRendering(self.offscreen)
        self.ren_win.AddRenderer(self.ren)
        
        # Initialize WindowToImageFilter
        self.w2if = vtk.vtkWindowToImageFilter()
        self.w2if.SetInput(self.ren_win)
        
        # Initialize WindowToDepthFilter
        self.wtdf = vtk.vtkImageShiftScale()
        self.wtdf.SetOutputScalarTypeToUnsignedChar()
        self.wtdf.SetInputConnection(self.w2if.GetOutputPort())
        self.wtdf.SetShift(0)
        self.wtdf.SetScale(-255)

    def random_transform(self, size=1):
        rx = np.random.randint(self.min_x_angle, self.max_x_angle, size=size)
        ry = np.random.randint(self.min_y_angle, self.max_y_angle, size=size)
        rz = np.random.randint(self.min_z_angle, self.max_z_angle, size=size)

        # the following values are currently not used
        scale = np.random.uniform(self.min_scale, self.max_scale, size=size)
        tx    = np.random.randint(self.min_tx, self.max_tx, size=size)
        ty    = np.random.randint(self.min_ty, self.max_ty, size=size)

        return np.stack((rx, ry, rz, scale, tx, ty), axis=1)

    # Generate nview 3D transformations and return them as a stack
    def generate_3d_transformations(self):
        if self.n_views == 9:
            return np.array(
                [
                    # [angle up down, angle left right, scale, tx, ty, tz]
                    # angle from above
                    [ 30,  15, 0, 0, 0, 0],
                    [ 30, -15, 0, 0, 0, 0],
                    [ 30,  45, 0, 0, 0, 0],
                    [ 30, -45, 0, 0, 0, 0],
                    # center view
                    [  0,   0, 0, 0, 0, 0],
                    # angle from below
                    [-30,  15, 0, 0, 0, 0],
                    [-30, -15, 0, 0, 0, 0],
                    [-30,  45, 0, 0, 0, 0],
                    [-30, -45, 0, 0, 0, 0],
                ], dtype=np.float32
            )
        return self.random_transform(size=self.n_views)

    def render_3d_multi_rgb_geometry_depth(self, transform_stack, file_name):
        tt = time.time()
        
        image_stack = np.empty((self.n_views, *self.image_size, 4), dtype=np.float32)
        actor, pd = obj_to_actor(file_name)
        self.ren.AddActor(actor)
        print("Render [1] - Setup time: ", f"{time.time() - tt:08.6f} s")

        t = vtk.vtkTransform()
        t.Identity()
        t.Update()
        
         # Transform (assuming only one mesh)
        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t)
        trans.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(trans.GetOutput())
        actor.SetMapper(mapper)
        
        tt = time.time()
        for i, (rx, ry, rz, *_) in enumerate(transform_stack):
            t.Identity()
            t.RotateY(ry)
            t.RotateX(rx)
            t.RotateZ(rz)
            t.Update()
            trans.Update()

            zmin = trans.GetOutput().GetBounds()[4]
            zmax = trans.GetOutput().GetBounds()[5]
            
            self.ren.GetActiveCamera().SetParallelScale(self.side_length)
            self.ren.GetActiveCamera().SetPosition(0, 0, 500)
            self.ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
            self.ren.GetActiveCamera().SetClippingRange(500 - zmax - self.slack, 500 - zmin + self.slack)

            mapper.Modified()
            self.ren.Modified()  # force actors to have the correct visibility
            self.ren_win.Render()

            self.w2if.SetInputBufferTypeToRGB()
            self.w2if.Modified()  # Needed here else only first rendering is put to file
            self.w2if.Update()

            # add rendering to image stack
            image_stack[i, :, :, 0:3] = vtk_to_numpy(self.w2if.GetOutput().GetPointData().GetScalars()).reshape(self.image_size[0], self.image_size[1], 3)

            self.ren.Modified()  # force actors to have the correct visibility
            self.w2if.SetInputBufferTypeToZBuffer()
            self.w2if.Modified()
            self.wtdf.Update()
            
            image_stack[i, :, :, 3:4] = vtk_to_numpy(self.wtdf.GetOutput().GetPointData().GetScalars()).reshape(self.image_size[0], self.image_size[1], 1)
            
        print('Render [2] - Render', f"{time.time() - tt:08.6f} s")
    
        # remove actors
        self.ren.RemoveActor(actor)
        # all the images in the stack are upside down, so we flip them 
        return np.flip(image_stack, axis=1), pd

    def multiview_render(self, file_name: Path):
        t = time.time()
        if not file_name.exists():
            raise FileNotFoundError(f"File {file_name} does not exist")
        if not file_name.is_file():
            raise FileNotFoundError(f"File {file_name} is not a file")
        if not file_name.suffix == ".obj":
            raise ValueError(f"File {file_name} is not an .obj file. Only .obj files are supported.")
        print('Render [0] - Prepare', f"{time.time() - t:08.6f} s")
        
        transformation_stack = self.generate_3d_transformations()
        image_stack, pd = self.render_3d_multi_rgb_geometry_depth(transformation_stack, file_name)
        image_stack = image_stack / 255

        return image_stack, transformation_stack, pd
