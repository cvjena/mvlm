__all__ = ["PaulsenPredictor"]

import math
import random
import time

import imageio
import matplotlib.pyplot as plt
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
    def __init__(self, config):
        super().__init__()
        self.config = config
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


    def find_heat_map_maxima(self, heatmaps, sigma=None, method="simple"):
        """ heatmaps: (#LM, hm_size,hm_size) """
        out_dim = heatmaps.shape[0]  # number of landmarks
        hm_size = heatmaps.shape[1]
        # coordinates = np.zeros((out_dim, 2), dtype=np.float32)
        coordinates = np.zeros((out_dim, 3), dtype=np.float32)

        # TODO Need to figure out why x and y are switched here...probably something with row, col
        # simple: Use only maximum pixel value in HM
        if method == "simple":
            for k in range(out_dim):
                highest_idx = np.unravel_index(np.argmax(heatmaps[k, :, :]), (hm_size, hm_size))
                px = highest_idx[0]
                py = highest_idx[1]
                value = heatmaps[k, px, py]  # TODO check if values is equal to np.max(hm)
                coordinates[k] = (px - 1, py - 0.5, value)  # TODO find out why it works with the subtractions

        if method == "moment":
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
            heatmap_maxima[:, idx, :] = self.find_heat_map_maxima(heatmaps[idx], method='simple')

    def generate_image_with_heatmap_maxima(self, image, heat_map):
        im_size = image.shape[0]
        hm_size = heat_map.shape[2]
        i = image.copy()

        coordinates = self.find_heat_map_maxima(heat_map, method='moment')

        # the predicted heat map is sometimes smaller than the input image
        factor = im_size / hm_size
        for c in range(coordinates.shape[0]):
            px = coordinates[c][0]
            py = coordinates[c][1]
            if not np.isnan(px) and not np.isnan(py):
                cx = int(px * factor)
                cy = int(py * factor)
                for x in range(cx - 2, cx + 2):
                    for y in range(cy - 2, cy + 2):
                        i[x, y, 0] = 0
                        i[x, y, 1] = 0
                        i[x, y, 2] = 1  # blue
        return i

    def show_image_and_heatmap(self, image, heat_map):
        heat_map = heat_map.numpy()
        # Generate combined heatmap image in RGB channels.
        # This must be possible to do smarter - Alas! My Python skillz are lacking
        hm = np.zeros((heat_map.shape[1], heat_map.shape[2], 3))
        n_lm = heat_map.shape[0]
        
        image_out = image[:, :, 0:3]
        
        for lm in range(n_lm):
            r = random.random()  # generate random colour placed on the unit sphere in RGB space
            g = random.random()
            b = random.random()
            length = math.sqrt(r * r + g * g + b * b)
            r = r / length
            g = g / length
            b = b / length
            hm[:, :, 0] = hm[:, :, 0] + heat_map[lm, :, :] * r
            hm[:, :, 1] = hm[:, :, 1] + heat_map[lm, :, :] * g
            hm[:, :, 2] = hm[:, :, 2] + heat_map[lm, :, :] * b

        im_marked = self.generate_image_with_heatmap_maxima(image_out, heat_map)

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(image_out)
        ax[1].imshow(hm)
        ax[2].imshow(im_marked)
        plt.show()

    def write_batch_of_heatmaps(self, heatmaps, images, cur_id):
        batch_size = heatmaps.shape[0]

        for idx in range(batch_size):
            name_hm_maxima = str(self.config.temp_dir / ('heatmap' + str(cur_id + idx) + '.png'))
            name_hm_maxima_2 = str(self.config.temp_dir / ('heatmap_max' + str(cur_id + idx) + '.png'))
            heatmap = heatmaps[idx, :, :, :]
            heatmap = heatmap.numpy()
            hm_size = heatmap.shape[2]

            hm = np.zeros((hm_size, hm_size, 3))
            n_lm = heatmap.shape[0]
            for lm in range(n_lm):
                r = random.random()  # generate random colour placed on the unit sphere in RGB space
                g = random.random()
                b = random.random()
                length = math.sqrt(r * r + g * g + b * b)
                r = r / length
                g = g / length
                b = b / length
                hm[:, :, 0] = hm[:, :, 0] + heatmap[lm, :, :] * r
                hm[:, :, 1] = hm[:, :, 1] + heatmap[lm, :, :] * g
                hm[:, :, 2] = hm[:, :, 2] + heatmap[lm, :, :] * b

            imageio.imwrite(name_hm_maxima, hm)

            im = images[idx]
            im_marked = self.generate_image_with_heatmap_maxima(im, heatmap)

            imageio.imwrite(name_hm_maxima_2, im_marked)

    def predict_landmarks_from_images(self, image_stack):
        # if cuda
        torch.set_float32_matmul_precision("medium")
        n_views = self.config['data_loader']['args']['n_views']
        batch_size = self.config['data_loader']['args']['batch_size']
        n_landmarks = self.config['arch']['args']['n_landmarks']
        heatmap_maxima = np.empty((n_landmarks, n_views, 3))
        # move all images to the GPU
        image_stack_d = torch.from_numpy(image_stack)
        image_stack_d = image_stack_d.to(self.device)
        image_stack_d = image_stack_d.permute(0, 3, 1, 2)  # from BHWC to BCHW
  
        heatmaps = torch.zeros((n_views, n_landmarks, 256, 256), device=self.device)
        cur_id = 0
        t = time.time()
        pre_times = []
        with torch.no_grad():
            while cur_id + batch_size <= n_views:
                cur_images = image_stack_d[cur_id:cur_id + batch_size, :, :, :]
                # print('predicting heatmaps for batch ', cur_id, ' to ', cur_id + batch_size)
                # data = data.to(self.device)
                tt = time.time()
                output = self.model(cur_images)
                pre_times.append(time.time() - tt)
                # output [stack (0 or 1), batch, lm, hm_size, hm_size]
                # heatmaps = output[1, :, :, :, :].cpu()
                heatmaps[cur_id:cur_id + batch_size, :, :, :] = output[1, :, :, :, :].squeeze(0)
                cur_id = cur_id + batch_size
        
        print("Prediction [0] - GPU time: ", self.p_time(time.time() - t))
        print("Prediction [0] - GPU time (mean): ", np.mean(pre_times))
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()        
        heatmaps = heatmaps.cpu()
        print("Prediction [1] - Copy to CPU: ", self.p_time(time.time() - t))
        
        # self.show_image_and_heatmap(image_stack[0], heatmaps[0])

        t = time.time()
        self.find_maxima_in_batch_of_heatmaps(heatmaps,  heatmap_maxima)
        print("Prediction [2] - Find maxima: ", self.p_time(time.time() - t))
        return heatmap_maxima
    
    def p_time(self, t):
        return f"{t:08.6f} s"