__all__ = ["ObjVTKRenderer3D"]
import math
import time
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from utils3d import obj_to_actor

class ObjVTKRenderer3D:
    def __init__(
        self,
        n_views: int = 8,
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
        if self.n_views == 8:
            return np.array(
                [
                    # [angle up down, angle left right, scale, tx, ty, tz]
                    # angle from above
                    [ 30,  15, 0, 0, 0, 0],
                    [ 30, -15, 0, 0, 0, 0],
                    [ 30,  45, 0, 0, 0, 0],
                    [ 30, -45, 0, 0, 0, 0],
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

    @staticmethod
    def get_landmark_bounds(lms):
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

    @staticmethod
    def get_landmarks_bounding_box_diagonal_length(lms):
        x_min, x_max, y_min, y_max, z_min, z_max = ObjVTKRenderer3D.get_landmark_bounds(lms)

        diag_len = math.sqrt(
            (x_max - x_min) * (x_max - x_min) + (y_max - y_min) * (y_max - y_min) + (z_max - z_min) * (z_max - z_min))
        return diag_len

    @staticmethod
    def get_landmarks_as_spheres(lms):
        diag_len = ObjVTKRenderer3D.get_landmarks_bounding_box_diagonal_length(lms)
        # sphere radius is 0.8% of bounding box diagonal
        sphere_size = diag_len * 0.008

        append = vtk.vtkAppendPolyData()
        for idx in range(len(lms)):
            lm = lms[idx]
            # scalars = vtk.vtkDoubleArray()
            # scalars.SetNumberOfComponents(1)

            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(lm)
            sphere.SetRadius(sphere_size)
            sphere.SetThetaResolution(20)
            sphere.SetPhiResolution(20)
            sphere.Update()
            # scalars.SetNumberOfValues(sphere.GetOutput().GetNumberOfPoints())

            # for s in range(sphere.GetOutput().GetNumberOfPoints()):
            #    scalars.SetValue(s, dst)

            # sphere.GetOutput().GetPointData().SetScalars(scalars)
            append.AddInputData(sphere.GetOutput())
            del sphere
            # del scalars

        append.Update()
        return append.GetOutput()

    # @staticmethod
    # def visualise_mesh_and_landmarks(mesh_name, landmarks=None):
    #     file_type = os.path.splitext(mesh_name)[1]
    #     win_size = 512

    #     ren = vtk.vtkRenderer()
    #     ren.SetBackground(1, 1, 1)

    #     # Initialize RenderWindow
    #     ren_win = vtk.vtkRenderWindow()
    #     ren_win.AddRenderer(ren)
    #     ren_win.SetSize(win_size, win_size)

    #     file_read = False
    #     if file_type == ".obj":
    #         mtl_name = os.path.splitext(mesh_name)[0] + '.mtl'
    #         # only do this for textured files
    #         if os.path.isfile(mtl_name):
    #             obj_dir = os.path.dirname(mesh_name)
    #             obj_in = vtk.vtkOBJImporter()
    #             obj_in.SetFileName(mesh_name)
    #             if os.path.isfile(mtl_name):
    #                 obj_in.SetFileNameMTL(mtl_name)
    #                 obj_in.SetTexturePath(obj_dir)
    #             obj_in.Update()

    #             obj_in.SetRenderWindow(ren_win)
    #             obj_in.Update()

    #             props = vtk.vtkProperty()
    #             props.SetColor(1, 1, 1)
    #             props.SetDiffuse(0)
    #             props.SetSpecular(0)
    #             props.SetAmbient(1)

    #             actors = ren.GetActors()
    #             actors.InitTraversal()
    #             actor = actors.GetNextItem()
    #             while actor:
    #                 actor.SetProperty(props)
    #                 actor = actors.GetNextItem()
    #             del props
    #             file_read = True

    #     if not file_read and file_type in [".vtk", ".stl", ".ply", ".wrl", ".obj"]:
    #         pd = Utils3D.multi_read_surface(mesh_name)
    #         if pd.GetNumberOfPoints() < 1:
    #             print('Could not read', mesh_name)
    #             return None

    #         texture_img = Utils3D.multi_read_texture(mesh_name)
    #         if texture_img is not None:
    #             pd.GetPointData().SetScalars(None)
    #             texture = vtk.vtkTexture()
    #             texture.SetInterpolate(1)
    #             texture.SetQualityTo32Bit()
    #             texture.SetInputData(texture_img)

    #         mapper = vtk.vtkPolyDataMapper()
    #         mapper.SetInputData(pd)

    #         actor_text = vtk.vtkActor()
    #         actor_text.SetMapper(mapper)
    #         if texture_img is not None:
    #             actor_text.SetTexture(texture)
    #             actor_text.GetProperty().SetColor(1, 1, 1)
    #             actor_text.GetProperty().SetAmbient(1.0)
    #             actor_text.GetProperty().SetSpecular(0)
    #             actor_text.GetProperty().SetDiffuse(0)
    #         ren.AddActor(actor_text)

    #     if landmarks is not None:
    #         lm_pd = Render3D.get_landmarks_as_spheres(landmarks)

    #         mapper = vtk.vtkPolyDataMapper()
    #         mapper.SetInputData(lm_pd)

    #         actor_lm = vtk.vtkActor()
    #         actor_lm.SetMapper(mapper)
    #         actor_lm.GetProperty().SetColor(0, 0, 1)
    #         ren.AddActor(actor_lm)

    #     # axes = vtk.vtkAxesActor()
    #     # ren.AddActor(axes)

    #     # ren.GetActiveCamera().SetPosition(0, 0, 1)
    #     # ren.GetActiveCamera().SetFocalPoint(0, 0, 0)
    #     # ren.GetActiveCamera().SetViewUp(0, 1, 0)
    #     # ren.GetActiveCamera().SetParallelProjection(1)

    #     iren = vtk.vtkRenderWindowInteractor()
    #     style = vtk.vtkInteractorStyleTrackballCamera()
    #     iren.SetInteractorStyle(style)
    #     iren.SetRenderWindow(ren_win)

    #     ren_win.Render()
    #     iren.Start()
