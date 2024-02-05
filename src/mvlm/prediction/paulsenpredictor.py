__all__ = ["PaulsenPredictor"]

from pathlib import Path
import mvlm.model.nnmodel as module_arch
import numpy as np
import torch
from torch.utils.model_zoo import load_url

from .predictor2d import Predictor2D

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


class PaulsenPredictor(Predictor2D):
    def __init__(
        self, 
        config,
        batch_size=2,
        selection_method="simple",
        # model parameters
        model_type: str = "MVLMModel_BU_3DFE",
        n_gpus = 1,
    ):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.selection_method = selection_method
        
        self.model_type = model_type
        self.n_gpus = n_gpus
      
        self.device, self.model = self._get_device_and_load_model_from_url()
        
    def get_lm_count(self) -> int:
        return 84
        
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
        torch_model = self.config.initialize('arch', module_arch)

        print('Loading checkpoint')
        model_dir = Path(__file__).parent / "models"  #self.config['trainer']['save_dir'] + "/trained/"
        model_indetification = self.model_type + '-' + "RGB+depth"
        
        check_point_name = models_urls[model_indetification]

        print('Getting device')
        device, device_ids = self._prepare_device(self.n_gpus)
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
            torch_model = torch.nn.DataParallel(torch_model, device_ids=device_ids)

        torch_model.load_state_dict(state_dict)
        torch_model = torch_model.to(device)
        torch_model.eval()
        return device, torch_model


    def find_heat_map_maxima(self, heatmaps):
        """ heatmaps: (#LM, hm_size,hm_size) """
        out_dim = heatmaps.shape[0]  # number of landmarks
        hm_size = heatmaps.shape[1]
        # coordinates = np.zeros((out_dim, 2), dtype=np.float32)
        coordinates = np.zeros((out_dim, 3), dtype=np.float32)

        # TODO Need to figure out why x and y are switched here...probably something with row, col
        # simple: Use only maximum pixel value in HM
        if self.selection_method == "simple":
            for k in range(out_dim):
                highest_idx = np.unravel_index(np.argmax(heatmaps[k, :, :]), (hm_size, hm_size))
                px = highest_idx[0]
                py = highest_idx[1]
                value = heatmaps[k, px, py]  # TODO check if values is equal to np.max(hm)
                coordinates[k] = (px - 1, py - 0.5, value)  # TODO find out why it works with the subtractions

        if self.selection_method == "moment":
            for k in range(out_dim):
                hm = heatmaps[k, :, :]
                highest_idx = np.unravel_index(np.argmax(hm), (hm_size, hm_size))
                px = highest_idx[0]
                py = highest_idx[1]

                value = np.max(hm)

                # Size of window around max (15 on each side gives an array of 2 * 5 + 1 values)
                sz = 15
                a_len = 2 * sz + 1
                if px > sz and hm_size-px > sz and py > sz and hm_size-py > sz:
                    slc = hm[px-sz:px+sz+1, py-sz:py+sz+1]
                    ar = np.arange(a_len)
                    sum_x = np.sum(slc, axis=1)
                    s = np.sum(np.multiply(ar, sum_x))
                    ss = np.sum(sum_x)
                    pos = s / ss - sz
                    px = px + pos

                    sum_y = np.sum(slc, axis=0)
                    s = np.sum(np.multiply(ar, sum_y))
                    ss = np.sum(sum_y)
                    pos = s / ss - sz
                    py = py + pos

                coordinates[k, :] = (px-1, py-0.5, value)  # TODO find out why it works with the subtractions

        return coordinates

    def find_maxima_in_batch_of_heatmaps(self, heatmaps, heatmap_maxima):
        heatmaps = heatmaps.numpy()
        batch_size = heatmaps.shape[0]
        for idx in range(batch_size):
            heatmap_maxima[:, idx, :] = self.find_heat_map_maxima(heatmaps[idx])
        return heatmap_maxima

    def predict_landmarks_from_images(self, image_stack: np.ndarray) -> np.ndarray:
        if self.device.type == 'cuda':
            torch.set_float32_matmul_precision("medium")
        n_views = image_stack.shape[0]
        n_landmarks = self.get_lm_count()
        
        heatmap_maxima = np.empty((n_landmarks, n_views, 3))
            
        # move all images to the GPU
        image_stack_d = torch.from_numpy(image_stack)
        image_stack_d = image_stack_d.to(self.device)
        image_stack_d = image_stack_d.permute(0, 3, 1, 2)  # from BHWC to BCHW
  
        heatmaps = torch.zeros((n_views, n_landmarks, 256, 256), device=self.device)
        cur_id = 0
        with torch.no_grad():
            while cur_id + self.batch_size <= n_views:
                cur_images = image_stack_d[cur_id:cur_id + self.batch_size, :, :, :]
                output = self.model(cur_images)
                heatmaps[cur_id:cur_id + self.batch_size, :, :, :] = output[1, :, :, :, :].squeeze(0)
                cur_id = cur_id + self.batch_size

        return self.find_maxima_in_batch_of_heatmaps(heatmaps.cpu(), heatmap_maxima)
