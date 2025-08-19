"""Ray and landmark visualization utilities using VTK.

Creates an interactive render window showing:
 - Surface mesh (polydata)
 - Rays (lines) from each view (line start/end pairs)
 - Estimated landmark intersection points (spheres)

Controls:
 - Mouse / trackball camera interaction (standard VTK)
 - Press 's' to save a high-resolution screenshot (PNG) to the screenshot output folder
 - Press 'q' or close window to exit

The screenshot magnification can be configured. Default is 2 (2x render size).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import vtk

from mvlm.utils.utils3d import obj_to_actor


@dataclass
class RayVisualizerConfig:
    window_size: tuple[int, int] = (900, 900)
    background: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ray_color: tuple[float, float, float] = (0.1, 0.4, 0.9)
    ray_opacity: float = 0.9
    landmark_color: tuple[float, float, float] = (0.9, 0.1, 0.1)
    landmark_radius: float = 2.0
    mesh_color: tuple[float, float, float] = (0.85, 0.85, 0.85)
    mesh_edge_color: tuple[float, float, float] = (0.3, 0.3, 0.3)
    mesh_opacity: float = 1.0
    screenshot_magnification: int = 2


class _KeyPressInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):  # noqa: D401
        super().__init__()
        self.parent = vtk.vtkRenderWindowInteractor() if parent is None else parent
        self._screenshot_dir: Path | None = None
        self._screenshot_magnification: int = 2
        self._window: vtk.vtkRenderWindow | None = None
        self._renderer: vtk.vtkRenderer | None = None

    def set_context(self, window: vtk.vtkRenderWindow, renderer: vtk.vtkRenderer, screenshot_dir: Path | None, mag: int):
        self._window = window
        self._renderer = renderer
        self._screenshot_dir = screenshot_dir
        self._screenshot_magnification = mag

    def OnKeyPress(self):  # noqa: N802 (VTK naming)
        key = self.GetInteractor().GetKeySym()
        if key.lower() == "s":
            print("[RayVisualizer] 's' detected -> capturing screenshot...")
            self._save_screenshot()
        elif key.lower() in ("q", "escape"):
            self.GetInteractor().GetRenderWindow().Finalize()
            self.GetInteractor().TerminateApp()
        else:
            super().OnKeyPress()

    def _save_screenshot(self):
        if self._window is None:
            print("[RayVisualizer] Cannot capture (no window).")
            return
        if self._screenshot_dir is None:
            print("[RayVisualizer] No screenshot directory provided.")
            return
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the latest frame is rendered
        self._window.Render()
        ts = time.strftime("%Y%m%d_%H%M%S")
        # Add basic camera info to filename for reproducibility
        cam = self._renderer.GetActiveCamera() if self._renderer else None
        if cam:
            pos = cam.GetPosition()
            fname = f"rayviz_{ts}_cam_{pos[0]:.1f}_{pos[1]:.1f}_{pos[2]:.1f}.png"
        else:
            fname = f"rayviz_{ts}.png"
        out_file = self._screenshot_dir / fname

        # Primary approach: WindowToImageFilter
        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(self._window)
        # Support both single int or (x,y) scaling depending on VTK version
        try:
            w2if.SetScale(self._screenshot_magnification, self._screenshot_magnification)
        except TypeError:  # older signature
            w2if.SetScale(self._screenshot_magnification)
        w2if.SetInputBufferTypeToRGBA()
        # For double-buffered windows: capture back buffer (front may be empty until swap)
        try:
            w2if.ReadFrontBufferOff()
        except AttributeError:
            pass
        w2if.Update()

        # Validate output dimensions; if zero, retry using RGB + front buffer
        dims = w2if.GetOutput().GetDimensions()
        if dims[0] == 0 or dims[1] == 0:
            print("[RayVisualizer] First capture empty, retrying with RGB / front buffer...")
            w2if.SetInputBufferTypeToRGB()
            try:
                w2if.ReadFrontBufferOn()
            except AttributeError:
                pass
            self._window.Render()
            w2if.Update()
            dims = w2if.GetOutput().GetDimensions()
            if dims[0] == 0 or dims[1] == 0:
                print("[RayVisualizer] Screenshot failed (zero-sized image).")
                return

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(out_file))
        writer.SetInputConnection(w2if.GetOutputPort())
        try:
            writer.Write()
            print(f"[RayVisualizer] Screenshot saved -> {out_file}  (size={dims[0]}x{dims[1]})")
        except Exception as e:  # noqa: BLE001
            print(f"[RayVisualizer] Failed to write screenshot: {e}")


class RayVisualizer:
    def __init__(self, config: RayVisualizerConfig | None = None):
        self.config = config or RayVisualizerConfig()

    def _create_mesh_actor(self, pd: vtk.vtkPolyData, obj_path: Path | None = None) -> vtk.vtkActor:
        # If original obj path provided, rebuild actor (including texture) from file
        if obj_path is not None and obj_path.exists():
            try:
                actor, _pd = obj_to_actor(obj_path)
                return actor
            except Exception as e:  # noqa: BLE001
                print(f"[RayVisualizer] Failed to create textured actor from {obj_path}: {e}. Falling back to plain actor.")
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*self.config.mesh_color)
        actor.GetProperty().SetOpacity(self.config.mesh_opacity)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(*self.config.mesh_edge_color)
        return actor

    def _clip_rays_to_mesh(self, starts: np.ndarray, ends: np.ndarray, mesh: vtk.vtkPolyData) -> np.ndarray:
        """Clip each ray segment to first intersection with mesh (if any)."""
        obb = vtk.vtkOBBTree()
        obb.SetDataSet(mesh)
        obb.BuildLocator()
        new_ends = ends.copy()
        n_landmarks, n_views, _ = starts.shape
        pts = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()
        for lm in range(n_landmarks):
            for v in range(n_views):
                p0 = starts[lm, v]
                p1 = ends[lm, v]
                pts.Reset()
                cell_ids.Reset()
                # IntersectWithLine returns number of intersection points (VTK variant differences)
                hit = obb.IntersectWithLine(p0, p1, pts, cell_ids)
                if hit and pts.GetNumberOfPoints() > 0:
                    ip = pts.GetPoint(0)
                    new_ends[lm, v] = ip
        return new_ends

    def _filter_indices(self, line_starts: np.ndarray, line_ends: np.ndarray, landmark_indices, view_indices):
        lm_idx = landmark_indices if landmark_indices is not None else list(range(line_starts.shape[0]))
        v_idx = view_indices if view_indices is not None else list(range(line_starts.shape[1]))
        fs = line_starts[np.ix_(lm_idx, v_idx)]
        fe = line_ends[np.ix_(lm_idx, v_idx)]
        return fs, fe, lm_idx, v_idx

    def _create_rays_actor(self, line_starts: np.ndarray, line_ends: np.ndarray) -> vtk.vtkActor:
        n_landmarks, n_views, _ = line_starts.shape
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for lm in range(n_landmarks):
            for v in range(n_views):
                p0 = line_starts[lm, v]
                p1 = line_ends[lm, v]
                id0 = points.InsertNextPoint(float(p0[0]), float(p0[1]), float(p0[2]))
                id1 = points.InsertNextPoint(float(p1[0]), float(p1[1]), float(p1[2]))
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, id0)
                line.GetPointIds().SetId(1, id1)
                lines.InsertNextCell(line)
        pd = vtk.vtkPolyData()
        pd.SetPoints(points)
        pd.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*self.config.ray_color)
        actor.GetProperty().SetOpacity(self.config.ray_opacity)
        actor.GetProperty().SetLineWidth(1.2)
        return actor

    def _create_landmark_actor(self, landmarks: np.ndarray) -> vtk.vtkActor:
        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        for lm in landmarks:
            pid = points.InsertNextPoint(float(lm[0]), float(lm[1]), float(lm[2]))
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)
        pd = vtk.vtkPolyData()
        pd.SetPoints(points)
        pd.SetVerts(verts)

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(self.config.landmark_radius)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(pd)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.ScalingOff()
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*self.config.landmark_color)
        return actor

    def show(
        self,
        mesh: vtk.vtkPolyData,
        line_starts: np.ndarray,
        line_ends: np.ndarray,
        landmarks: np.ndarray | None = None,
        screenshot_dir: Path | None = None,
        landmark_indices: list[int] | None = None,
        view_indices: list[int] | None = None,
        obj_path: Path | None = None,
        clip_to_mesh: bool = True,
    ):
        """Render interactive visualization.

        Parameters
        ----------
        mesh : vtkPolyData
            The original mesh polydata.
        line_starts, line_ends : np.ndarray
            Arrays shaped (n_landmarks, n_views, 3)
        landmarks : np.ndarray | None
            Final landmark positions shaped (n_landmarks, 3)
        screenshot_dir : Path | None
            Folder to store screenshots when 's' is pressed.
        """
        # --- Renderer and actors ---
        ren = vtk.vtkRenderer()
        ren.SetBackground(*self.config.background)

        # Filter indices if provided
        f_starts, f_ends, used_lm_idx, used_v_idx = self._filter_indices(line_starts, line_ends, landmark_indices, view_indices)
        if clip_to_mesh:
            f_ends = self._clip_rays_to_mesh(f_starts, f_ends, mesh)

        ren.AddActor(self._create_mesh_actor(mesh, obj_path=obj_path))
        ren.AddActor(self._create_rays_actor(f_starts, f_ends))
        if landmarks is not None:
            # Reduce landmarks if subset used
            if landmark_indices is not None:
                try:
                    landmarks_to_show = landmarks[used_lm_idx]
                except Exception:  # noqa: BLE001
                    landmarks_to_show = landmarks
            else:
                landmarks_to_show = landmarks
            ren.AddActor(self._create_landmark_actor(landmarks_to_show))

        # Outline (optional for scale reference)
        bounds = mesh.GetBounds()
        outline = vtk.vtkOutlineSource()
        outline.SetBounds(bounds)
        outline_mapper = vtk.vtkPolyDataMapper()
        outline_mapper.SetInputConnection(outline.GetOutputPort())
        outline_actor = vtk.vtkActor()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        outline_actor.GetProperty().SetOpacity(0.1)
        ren.AddActor(outline_actor)

        # --- Render window ---
        rw = vtk.vtkRenderWindow()
        rw.SetSize(*self.config.window_size)
        rw.AddRenderer(ren)
        rw.SetOffScreenRendering(False)

        # --- Interactor & style ---
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(rw)
        style = _KeyPressInteractorStyle()
        style.set_context(rw, ren, screenshot_dir, self.config.screenshot_magnification)
        iren.SetInteractorStyle(style)

        # --- Camera setup & initial render ---
        ren.ResetCamera()
        rw.Render()
        print("[RayVisualizer] Starting interactive session. Press 's' to save screenshot, 'q' to quit.")
        iren.Initialize()
        iren.Start()
