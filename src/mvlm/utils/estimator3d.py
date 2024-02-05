__all__ = ["Estimator3D"]

import numpy as np
import vtk

from .utils3d import compute_intersection_between_lines

def rotation_matrix_x(angle):
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

def rotation_matrix_y(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

def rotation_matrix_z(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


class Estimator3D:
    def __init__(
        self, 
        mode: str = "quantile",
        threshold_quantile: float = 0.5,
        threshold_abs: float = 0.5,
    ):
        self.mode = mode
        self.threshold_quantile = threshold_quantile
        self.threshold_abs = threshold_abs

    # Each maxima in a heatmap corresponds to a line in 3D space of the original 3D shape
    # This function transforms the maxima to (start point, end point) pairs
    def estimate_landmark_lines(
        self,
        image_stack: np.ndarray,
        landmarks_stack: np.ndarray, 
        transform_stack: np.ndarray
    ):
        n_landmarks = landmarks_stack.shape[0]
        n_views = landmarks_stack.shape[1]

        line_starts = np.empty((n_landmarks, n_views, 3))
        line_finish = np.empty((n_landmarks, n_views, 3))

        img_size = image_stack.shape[1]
        hm_size = image_stack.shape[1]

        # TODO these fixed values should probably be in a config file
        x_min = -150
        x_max = 150
        y_min = -150
        y_max = 150
        x_len = x_max - x_min
        y_len = y_max - y_min

        for idx in range(n_views):
            rx, ry, rz = transform_stack[idx, :3]
            t = np.diag(np.ones(4))
            t[0:3, 0:3] = (rotation_matrix_y(np.deg2rad(ry)) @ rotation_matrix_x(np.deg2rad(rx))) @ rotation_matrix_z(np.deg2rad(rz))

            for lm_no in range(n_landmarks):
                # [n_landmarks, n_views, x, y, value]
                y = landmarks_stack[lm_no, idx, 0]
                x = landmarks_stack[lm_no, idx, 1]
                # value = self.heatmap_maxima[lm_no, idx, 2]

                #  Extract just one landmark and scale it according to heatmap and image sizes
                y = y / hm_size * img_size
                x = x / hm_size * img_size

                # Making end points of line in world coordinates
                p_wc_s = np.zeros((3, 1))
                p_wc_e = np.zeros((3, 1))

                p_wc_s[0] = (x / img_size) * x_len + x_min
                p_wc_s[1] = ((img_size - 1 - y) / img_size) * y_len + y_min
                p_wc_s[2] = 500

                p_wc_e[0] = (x / img_size) * x_len + x_min
                p_wc_e[1] = ((img_size - 1 - y) / img_size) * y_len + y_min
                p_wc_e[2] = -500
                
                points = np.concatenate((p_wc_s, p_wc_e), axis=1)
                points = np.concatenate((points, np.ones((1, 2))), axis=0)
                points = np.matmul(t.T, points)
                p_wc_s = points[:, 0]
                p_wc_e = points[:, 1]
                
                line_starts[lm_no, idx, :] = p_wc_s[:3]
                line_finish[lm_no, idx, :] = p_wc_e[:3]

        return line_starts, line_finish 

    def compute_intersection_between_lines_ransac(self, pa, pb):
        # TODO parameters in config
        # iterations = 1
        best_error = 100000000  # TODO should find a better initialiser
        best_p = (0, 0, 0)
        dist_thres = 10 * 10  # TODO should find a better way to esimtate dist_thres
        # d = 10  #
        n_lines = len(pa)
        d = n_lines / 3
        used_lines = -1

        # for i in range(iterations):
        # get 3 random lines
        ran_lines = np.random.choice(range(n_lines), 8, replace=True)
        # Compute first estimate of intersection
        p_est = compute_intersection_between_lines(pa[ran_lines, :], pb[ran_lines, :])
        # Compute distance from all lines to intersection
        top = np.cross((np.transpose(p_est) - pa), (np.transpose(p_est) - pb))
        bottom = pb - pa
        distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1))**2
        # number of inliners
        n_inliners = np.sum(distances < dist_thres)
        if n_inliners > d:
            # reestimate based on inliners
            idx = distances < dist_thres
            p_est = compute_intersection_between_lines(pa[idx, :], pb[idx, :])

            # Compute distance from all inliners to intersection
            top = np.cross((np.transpose(p_est) - pa[idx, :]), (np.transpose(p_est) - pb[idx, :]))
            bottom = pb[idx, :] - pa[idx, :]
            distances = (np.linalg.norm(top, axis=1) / np.linalg.norm(bottom, axis=1))**2

            # sum_squared = np.sum(np.square(distances)) / n_inliners
            sum_squared = np.sum(distances) / n_inliners
            if sum_squared < best_error:
                best_error = sum_squared
                best_p = p_est
                used_lines = n_inliners

        if used_lines == -1:
            # self.logger.warning('Ransac failed - estimating from all lines')
            best_p = compute_intersection_between_lines(pa, pb)
        # else:
        # print('Ransac error ', best_error, ' with ', used_lines, ' lines')

        return best_p, best_error

    # return the lines that correspond to a high valued maxima in the heatmap
    def filter_lines_based_on_heatmap_value_using_quantiles(self, landmark_stack, lm_no, pa, pb):
        max_values = landmark_stack[lm_no, :, 2]
        threshold = np.quantile(max_values, self.threshold_quantile)
        idx = max_values > threshold
        # print('Using ', threshold, ' as threshold in heatmap maxima')
        pa_new = pa[idx]
        pb_new = pb[idx]
        return pa_new, pb_new

    # return the lines that correspond to a high valued maxima in the heatmap
    def filter_lines_based_on_heatmap_value_using_absolute_value(self, landmark_stack, lm_no, pa, pb):
        max_values = landmark_stack[lm_no, :, 2]
        idx = max_values > self.threshold_abs
        pa_new = pa[idx]
        pb_new = pb[idx]
        return pa_new, pb_new

    # Each landmark can be computed by the intersection of the view lines going trough (or near) it
    def estimate_landmarks_from_lines(self, landmark_stack, lines_s, lines_e):
        n_landmarks = lines_s.shape[0]
        landmarks = np.empty((n_landmarks, 3))
        sum_error = 0
        for lm_no in range(n_landmarks):
            pa = lines_s[lm_no, :, :]
            pb = lines_e[lm_no, :, :]
            
            if self.mode == "absolute":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_absolute_value(landmark_stack, lm_no, pa, pb)
            elif self.mode == "quantile":
                pa, pb = self.filter_lines_based_on_heatmap_value_using_quantiles(landmark_stack, lm_no, pa, pb)
            else:
                raise ValueError(f"Unknown mode for line matching in Estimator: {self.mode}")
            
            p_intersect = (0, 0, 0)
            if len(pa) < 3:
                raise Exception("Not enough valid view lines for landmark lm_no")
            
            p_intersect, best_error = self.compute_intersection_between_lines_ransac(pa, pb)
            sum_error = sum_error + best_error
            landmarks[lm_no, :] = p_intersect
        # print("Ransac average error ", sum_error/n_landmarks)
        return landmarks, sum_error/n_landmarks

    # TODO this is also present in render3D - should probably be merged
    # def apply_pre_transformation(self, pd):
    #     translation = [0, 0, 0]
    #     if self.config['pre-align']['align_center_of_mass']:
    #         vtk_cm = vtk.vtkCenterOfMass()
    #         vtk_cm.SetInputData(pd)
    #         vtk_cm.SetUseScalarsAsWeights(False)
    #         vtk_cm.Update()
    #         cm = vtk_cm.GetCenter()
    #         translation = [-cm[0], -cm[1], -cm[2]]

    #     t = vtk.vtkTransform()
    #     t.Identity()

    #     rx = self.config['pre-align']['rot_x']
    #     ry = self.config['pre-align']['rot_y']
    #     rz = self.config['pre-align']['rot_z']
    #     s = self.config['pre-align']['scale']

    #     t.Scale(s, s, s)
    #     t.RotateY(ry)
    #     t.RotateX(rx)
    #     t.RotateZ(rz)
    #     t.Translate(translation)
    #     t.Update()

    #     # Transform (assuming only one mesh)
    #     trans = vtk.vtkTransformPolyDataFilter()
    #     trans.SetInputData(pd)
    #     trans.SetTransform(t)
    #     trans.Update()

    #     if self.config['pre-align']['write_pre_aligned']:
    #         name_out = str(self.config.temp_dir / ('pre_transform_mesh.vtk'))
    #         writer = vtk.vtkPolyDataWriter()
    #         writer.SetInputData(trans.GetOutput())
    #         writer.SetFileName(name_out)
    #         writer.Write()

    #     return trans.GetOutput(), t

    def transform_landmarks_to_original_space(self, landmarks, t):
        points = vtk.vtkPoints()
        pd = vtk.vtkPolyData()
        # verts = vtk.vtkCellArray()

        for lm in landmarks:
            pid = points.InsertNextPoint(lm)
            # verts.InsertNextCell(1)
            # verts.InsertCellPoint(pid)
        pd.SetPoints(points)

        trans = vtk.vtkTransformPolyDataFilter()
        trans.SetInputData(pd)
        trans.SetTransform(t.GetInverse())
        trans.Update()
        pd_trans = trans.GetOutput()

        n_landmarks = pd_trans.GetNumberOfPoints()
        new_landmarks = np.zeros((n_landmarks, 3))
        for lm_no in range(pd_trans.GetNumberOfPoints()):
            p = pd_trans.GetPoint(lm_no)
            new_landmarks[lm_no, :] = (p[0], p[1], p[2])
        return new_landmarks

    # Project found landmarks to closest point on the target surface
    # return the landmarks in the original space
    def project_landmarks_to_surface(self, pd, landmarks):
        # TODO the projection should be fixed such that the mesh has its center of mass in the origin
        #      but then the landmarks should be transformed back to the original space
        
        # pd = self.multi_read_surface(mesh_name)
        # pd, t = self.apply_pre_transformation(pd)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputData(pd)
        # # clean.SetInputConnection(pd.GetOutputPort())
        clean.Update()

        locator = vtk.vtkCellLocator()
        locator.SetDataSet(clean.GetOutput())
        locator.SetNumberOfCellsPerBucket(1)
        locator.BuildLocator()

        projected_landmarks = np.copy(landmarks)
        n_landmarks = landmarks.shape[0]

        for i in range(n_landmarks):
            p = landmarks[i, :]
            cell_id = vtk.mutable(0)
            sub_id = vtk.mutable(0)
            dist2 = vtk.reference(0)
            tcp = np.zeros(3)

            locator.FindClosestPoint(p, tcp, cell_id, sub_id, dist2)
            # print('Nearest point in distance ', np.sqrt(np.float(dist2)))
            projected_landmarks[i, :] = tcp

        del clean
        del locator
        # self.landmarks = projected_landmarks
        return projected_landmarks
        # return self.transform_landmarks_to_original_space(projected_landmarks, t)

