import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.model_zoo import load_url

import model.model as module_arch
from prediction import Predict2D
from utils3d import ObjVTKRenderer3D, Utils3D

models_urls = {
    'MVLMModel_DTU3D-RGB':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB_07092019_only_state_dict-c0255a70.pth',
    'MVLMModel_DTU3D-depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_Depth_19092019_only_state_dict-95b89b63.pth',
    'MVLMModel_DTU3D-geometry':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry_only_state_dict-41851074.pth',
    'MVLMModel_DTU3D-geometry+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry+depth_20102019_15epoch_only_state_dict-73b20e31.pth',
    'MVLMModel_DTU3D-RGB+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB+depth_20092019_only_state_dict-e3c12463a9.pth',
    'MVLMModel_BU_3DFE-RGB':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB_24092019_6epoch_only_state_dict-eb652074.pth',
    'MVLMModel_BU_3DFE-depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_depth_10102019_4epoch_only_state_dict-e2318093.pth',
    'MVLMModel_BU_3DFE-geometry':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry_02102019_4epoch-only_state_dict-f85518fa.pth',
    'MVLMModel_BU_3DFE-RGB+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB+depth_05102019_5epoch_only_state_dict-297955f6.pth',
    'MVLMModel_BU_3DFE-geometry+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry+depth_17102019_13epoch_only_state_dict-aa34a6d68.pth'
  }


models_urls_full = {
    'MVLMModel_DTU3D-RGB':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB_07092019-c1cc3d59.pth',
    'MVLMModel_DTU3D-depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_Depth_19092019-ad636c81.pth',
    'MVLMModel_DTU3D-geometry':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry-9d2feee6.pth',
    'MVLMModel_DTU3D-geometry+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry+depth_20102019_15epoch-c2388595.pth',
    'MVLMModel_DTU3D-RGB+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB+depth_20092019-7fc1d845.pth',
    'MVLMModel_BU_3DFE-RGB':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB_24092019_6epoch-9f242c87.pth',
    'MVLMModel_BU_3DFE-depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_depth_10102019_4epoch-03b2f7b9.pth',
    'MVLMModel_BU_3DFE-geometry':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry_02102019_4epoch-052ee4b0.pth',
    'MVLMModel_BU_3DFE-RGB+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB+depth_05102019_5epoch-90e29350.pth',
    'MVLMModel_BU_3DFE-geometry+depth':
        'https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry+depth_17102019_13epoch-eb18dce4.pth'
  }

class TimeMixin:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        
    def tic(self):
        self.start_time = time.time()
        
    def toc(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

    def toc_p(self):
        return self.p_time(self.toc())

    def p_time(self, t):
        return f"{t:08.6f} s"
    


class DeepMVLM(TimeMixin):
    def __init__(
        self, 
        config: str, # basically the path to the config file
        render_image_stack: bool = False, # if true, the image stack will be saved
        render_image_folder: Path = None, # if not None, the image stack will be saved in this folder
    ):
        self.config = config
        self.render_image_stack = render_image_stack
        self.render_image_folder = render_image_folder
        
        # loading of the model
        self.device, self.model = self._get_device_and_load_model_from_url()
        
        # loading of the renderer and the predictor
        self.renderer_3d = ObjVTKRenderer3D()
        self.predictor_2d = Predict2D(self.config, self.model, self.device)
        self.estimator_3d = Utils3D(self.config)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        if n_gpu_use > 0 and torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] < 35):
            n_gpu_use = 0
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _get_device_and_load_model_from_url(self):
        print('Initialising model')
        model = self.config.initialize('arch', module_arch)

        print('Loading checkpoint')
        model_dir = self.config['trainer']['save_dir'] + "/trained/"
        model_name = self.config['name']
        image_channels = self.config['data_loader']['args']['image_channels']
        name_channels = model_name + '-' + image_channels
        check_point_name = models_urls[name_channels]

        print('Getting device')
        device, device_ids = self._prepare_device(self.config['n_gpu'])
        checkpoint = load_url(check_point_name, model_dir, map_location=device)

        # Write clean model - should only be done once for translation of models
        # base_name = os.path.basename(os.path.splitext(check_point_name)[0])
        # clean_file = 'saved/trained/' + base_name + '_only_state_dict.pth'
        # torch.save(checkpoint['state_dict'], clean_file)

        state_dict = []
        # Hack until all dicts are transformed
        if check_point_name.find('only_state_dict') == -1:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return device, model

    def predict_one_file(self, file_name: Path):
        full_s = time.time()
        if not file_name.exists():
            print(f"File {file_name} does not exist")
            return None
        self.tic()
        image_stack, transform_stack, pd = self.renderer_3d.multiview_render(file_name)
        print('Render [Total]: ', self.toc_p())
        
        if self.render_image_stack:
            # TODO Check if the folder exists
            self.visualize_image_stack(image_stack, file_name)
       
        self.tic()
        heatmap_maxima = self.predictor_2d.predict_heatmaps_from_images(image_stack)
        print('Prediction [Total]: ', self.toc_p())

        self.tic()
        self.estimator_3d.heatmap_maxima = heatmap_maxima
        self.estimator_3d.transformations_3d = transform_stack
        self.estimator_3d.compute_lines_from_heatmap_maxima()
        print('Landmarks [0] - From Heatmaps: ', self.toc_p())

        self.tic()
        error = self.estimator_3d.compute_all_landmarks_from_view_lines()
        print('Landmarks [1] - From View Lines: ', self.toc_p())

        self.tic()
        self.estimator_3d.project_landmarks_to_surface(pd)
        print('Landmarks [2] - Project to Surface: ', self.toc_p())
        print('Landmarks [Error]: ', f"{error:08.6f}", " mm")

        print("Landmarks 3D Total: ", self.p_time(time.time() - full_s))
        return self.estimator_3d.landmarks
    
    def visualize_image_stack(self, image_stack: np.ndarray, file_name: str):
         # save the image stack, make the in a 2 by 4 grid
        n, h, w, c = image_stack.shape
        out_image = np.reshape(image_stack, (2, 4, h, w, c))
        # remove last channel
        out_image = out_image[:, :, :, :, 0:3]
        out_image = np.transpose(out_image, (0, 2, 1, 3, 4))
        out_image = np.reshape(out_image, (2*h, 4*w, 3))
        out_image = np.uint8(out_image * 255)
        out_image = Image.fromarray(out_image)
        
        file_name = Path(file_name)
        out_image.save(f'{self.render_image_folder}/{file_name.stem}_{n}.png')

    @staticmethod
    def write_landmarks_as_vtk_points(landmarks, file_name):
        Utils3D.write_landmarks_as_vtk_points_external(landmarks, file_name)

    @staticmethod
    def write_landmarks_as_text(landmarks, file_name):
        Utils3D.write_landmarks_as_text_external(landmarks, file_name)

    @staticmethod
    def visualise_mesh_and_landmarks(mesh_name, landmarks=None):
        ObjVTKRenderer3D.visualise_mesh_and_landmarks(mesh_name, landmarks)# 
