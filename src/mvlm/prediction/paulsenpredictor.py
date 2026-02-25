__all__ = ["BU3DFEPredictor", "DTU3DPredictor"]

import abc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.hub import load_state_dict_from_url

from .predictor2d import Predictor2D

models_urls = {
    "MVLMModel_DTU3D-RGB": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB_07092019_only_state_dict-c0255a70.pth",
    "MVLMModel_DTU3D-depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_Depth_19092019_only_state_dict-95b89b63.pth",
    "MVLMModel_DTU3D-geometry": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry_only_state_dict-41851074.pth",
    "MVLMModel_DTU3D-geometry+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry+depth_20102019_15epoch_only_state_dict-73b20e31.pth",
    "MVLMModel_DTU3D-RGB+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB+depth_20092019_only_state_dict-e3c12463a9.pth",
    "MVLMModel_BU_3DFE-RGB": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB_24092019_6epoch_only_state_dict-eb652074.pth",
    "MVLMModel_BU_3DFE-depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_depth_10102019_4epoch_only_state_dict-e2318093.pth",
    "MVLMModel_BU_3DFE-geometry": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry_02102019_4epoch-only_state_dict-f85518fa.pth",
    "MVLMModel_BU_3DFE-RGB+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB+depth_05102019_5epoch_only_state_dict-297955f6.pth",
    "MVLMModel_BU_3DFE-geometry+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry+depth_17102019_13epoch_only_state_dict-aa34a6d68.pth",
}

models_urls_full = {
    "MVLMModel_DTU3D-RGB": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB_07092019-c1cc3d59.pth",
    "MVLMModel_DTU3D-depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_Depth_19092019-ad636c81.pth",
    "MVLMModel_DTU3D-geometry": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry-9d2feee6.pth",
    "MVLMModel_DTU3D-geometry+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_geometry+depth_20102019_15epoch-c2388595.pth",
    "MVLMModel_DTU3D-RGB+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_DTU3D_RGB+depth_20092019-7fc1d845.pth",
    "MVLMModel_BU_3DFE-RGB": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB_24092019_6epoch-9f242c87.pth",
    "MVLMModel_BU_3DFE-depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_depth_10102019_4epoch-03b2f7b9.pth",
    "MVLMModel_BU_3DFE-geometry": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry_02102019_4epoch-052ee4b0.pth",
    "MVLMModel_BU_3DFE-RGB+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_RGB+depth_05102019_5epoch-90e29350.pth",
    "MVLMModel_BU_3DFE-geometry+depth": "https://shapeml.compute.dtu.dk/Deep-MVLM/models/MVLMModel_BU_3DFE_geometry+depth_17102019_13epoch-eb18dce4.pth",
}


class PaulsenModel(Predictor2D):
    """
    BUG: Please note that these models work really well withing the 3D rendering pipeline, but fail quite stronly if used on loaded images.
         This indicates that somewhere the models cannot handle compression artifacts or other image quality issues, and are quite overfit on this data.
         Not a problem really for us here, but should be noted for future work.
    """

    def __init__(
        self,
        model_type: str,
        image_mode: str,
        n_gpus=1,
        batch_size=2,
        selection_method="simple",
        # model parameters
    ):
        super().__init__()
        self.batch_size = batch_size
        self.selection_method = selection_method

        self.model_type = model_type
        self.image_mode = image_mode
        self.n_gpus = n_gpus

        self.device, self.model = self._get_device_and_load_model_from_url()

    @abc.abstractmethod
    def get_lm_count(self) -> int:
        pass

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            n_gpu_use = n_gpu
        if n_gpu_use > 0 and torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] < 35):
            n_gpu_use = 0

        if not torch.cuda.is_available():
            print("No GPU available, using CPU.")
            n_gpu_use = 0

        device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _get_device_and_load_model_from_url(self):
        print("Initialising model")
        torch_model = MVLMModel(n_landmarks=self.get_lm_count(), n_features=256, dropout_rate=0.2, image_channels=self.image_mode)

        print("Loading checkpoint")
        model_dir = Path(__file__).parent / "models"
        model_indetification = self.model_type + "-" + self.image_mode

        check_point_name = models_urls[model_indetification]

        print("Getting device")
        device, device_ids = self._prepare_device(self.n_gpus)
        checkpoint = load_state_dict_from_url(check_point_name, model_dir=str(model_dir), map_location=device)
        state_dict = checkpoint["state_dict"] if check_point_name.find("only_state_dict") == -1 else checkpoint

        if len(device_ids) > 1:
            torch_model = torch.nn.DataParallel(torch_model, device_ids=device_ids)

        torch_model.load_state_dict(state_dict)
        torch_model = torch_model.to(device)
        torch_model.eval()
        return device, torch_model

    def find_heat_map_maxima(self, heatmaps):
        """heatmaps: (#LM, hm_size,hm_size)"""
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
                if px > sz and hm_size - px > sz and py > sz and hm_size - py > sz:
                    slc = hm[px - sz : px + sz + 1, py - sz : py + sz + 1]
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

                coordinates[k, :] = (px - 1, py - 0.5, value)  # TODO find out why it works with the subtractions

        return coordinates

    def find_maxima_in_batch_of_heatmaps(self, heatmaps, heatmap_maxima):
        heatmaps = heatmaps.numpy()
        batch_size = heatmaps.shape[0]
        for idx in range(batch_size):
            heatmap_maxima[:, idx, :] = self.find_heat_map_maxima(heatmaps[idx])
        return heatmap_maxima

    def predict_landmarks_from_images(self, image_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict landmarks for each view in image_stack.

        Fixes original batching bug where a remainder (or single view when batch_size>n_views)
        was never forwarded through the network, yielding zero heatmaps and constant
        (-1, -0.5, value) coordinates after maxima extraction.
        """

        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("medium")
        n_views = image_stack.shape[0]
        n_landmarks = self.get_lm_count()

        heatmap_maxima = np.empty((n_landmarks, n_views, 3), dtype=np.float32)
        valid = np.ones((n_views), dtype=bool)

        # move all images to device (expects values already in correct range 0..1)
        image_stack_d = torch.from_numpy(image_stack).to(self.device)
        image_stack_d = image_stack_d.permute(0, 3, 1, 2)  # BHWC -> BCHW

        heatmaps = torch.zeros((n_views, n_landmarks, 256, 256), device=self.device)
        cur_id = 0
        with torch.no_grad():
            while cur_id < n_views:  # process all, including last remainder
                end_id = min(cur_id + self.batch_size, n_views)
                cur_images = image_stack_d[cur_id:end_id]
                if cur_images.shape[0] == 0:
                    break
                output = self.model(cur_images)

                # Support different output container shapes
                if isinstance(output, (list, tuple)):
                    out_tensor = output[1]
                else:
                    out_tensor = output

                # If shape is (stages, B, C, H, W) choose final stage
                if out_tensor.dim() == 5:
                    out_tensor = out_tensor[-1]

                if out_tensor.dim() != 4:
                    raise RuntimeError(f"Unexpected heatmap tensor shape: {out_tensor.shape}")

                b = out_tensor.shape[0]
                heatmaps[cur_id : cur_id + b] = out_tensor[:b]
                cur_id = end_id

        print(heatmaps.shape, heatmaps.min(), heatmaps.max(), heatmaps.dtype)

        lms = self.find_maxima_in_batch_of_heatmaps(heatmaps.detach().cpu(), heatmap_maxima)
        return lms, valid


class BU3DFEPredictor(PaulsenModel):
    def __init__(
        self,
        batch_size=2,
        selection_method="simple",
        n_gpus=1,
    ):
        super().__init__(model_type="MVLMModel_BU_3DFE", image_mode="RGB+depth", n_gpus=n_gpus, batch_size=batch_size, selection_method=selection_method)

    def get_lm_count(self) -> int:
        return 84


class DTU3DPredictor(PaulsenModel):
    def __init__(
        self,
        batch_size=2,
        selection_method="simple",
        n_gpus=1,
    ):
        super().__init__(model_type="MVLMModel_DTU3D", image_mode="RGB+depth", n_gpus=n_gpus, batch_size=batch_size, selection_method=selection_method)

    def get_lm_count(self) -> int:
        return 73


### Actural Torch Model ###


# Residual block
# Inspired from https://github.com/1adrianb/face-alignment
class ResidualBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes // 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = nn.Conv2d(out_planes // 2, out_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = nn.Conv2d(out_planes // 4, out_planes // 4, kernel_size=3, stride=1, padding=1, bias=False)

        self.resample = nn.Identity()
        if in_planes != out_planes:
            self.resample = nn.Sequential(nn.BatchNorm2d(in_planes), nn.ReLU(True), nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        residual = x
        out1 = self.conv1(F.relu(self.bn1(x), True))
        out2 = self.conv2(F.relu(self.bn2(out1), True))
        out3 = self.conv3(F.relu(self.bn3(out2), True))
        residual = self.resample(residual)
        return torch.cat((out1, out2, out3), 1) + residual


class HourGlassModule(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = num_features
        self.rb1 = ResidualBlock(self.features, self.features)
        self.rb2 = ResidualBlock(self.features, self.features)
        self.rb3 = ResidualBlock(self.features, self.features)
        self.rb4 = ResidualBlock(self.features, self.features)
        self.rb5 = ResidualBlock(self.features, self.features)
        self.rb6 = ResidualBlock(self.features, self.features)
        self.rb7 = ResidualBlock(self.features, self.features)
        self.rb8 = ResidualBlock(self.features, self.features)
        self.rb9 = ResidualBlock(self.features, self.features)
        self.rb10 = ResidualBlock(self.features, self.features)
        self.rb11 = ResidualBlock(self.features, self.features)
        self.rb12 = ResidualBlock(self.features, self.features)
        self.rb13 = ResidualBlock(self.features, self.features)
        self.rb14 = ResidualBlock(self.features, self.features)
        self.rb15 = ResidualBlock(self.features, self.features)
        self.rb16 = ResidualBlock(self.features, self.features)
        self.rb17 = ResidualBlock(self.features, self.features)
        self.rb18 = ResidualBlock(self.features, self.features)
        self.rb19 = ResidualBlock(self.features, self.features)
        self.rb20 = ResidualBlock(self.features, self.features)

    def forward(self, x):
        # example input data
        # x : (128 x 128 x 256)
        # self.features = 256

        # Upper branch
        up1 = self.rb1(x)  # up1 (128 x 128 x 256)
        lowt1 = F.max_pool2d(x, 2)  # lowt1 (64 x 64 x 256)
        low1 = self.rb2(lowt1)  # low1 (64 x 64 x 256)

        # recursion (from org code): num_down_sample = 3
        up11 = self.rb3(low1)  # up11 (64 x 64 x 256)
        lowt11 = F.max_pool2d(low1, 2)  # lowt11 (32 x 32 x 256)
        low11 = self.rb4(lowt11)  # low11 (32 x 32 x 256)

        # recursion (from org code): num_down_sample = 2
        up12 = self.rb5(low11)  # up12 (32 x 32 x 256)
        lowt12 = F.max_pool2d(low11, 2)  # lowt12 (16 x 16 x 256)
        low12 = self.rb6(lowt12)  # low12 (16 x 16 x 256)

        # recursion (from org code): num_down_sample = 1
        up13 = self.rb7(low12)  # up13 (16 x 16 x 256)
        lowt13 = F.max_pool2d(low12, 2)  # lowt13 (8 x 8 x 256)
        low13 = self.rb8(lowt13)  # low13 (8 x 8 x 256)

        # recursion (from org code): num_down_sample = 0
        up14 = self.rb9(low13)  # up14 (8 x 8 x 256)
        lowt14 = F.max_pool2d(low13, 2)  # lowt14 (4 x 4 x 256)
        low14 = self.rb10(lowt14)  # low13 (4 x 4 x 256)

        # This is the bottleneck
        low2 = self.rb11(low14)  # low2 (4 x 4 x 256)
        low3 = self.rb12(low2)  # low3 (4 x 4 x 256)
        up2 = F.interpolate(low3, scale_factor=2, mode="nearest")  # up2 (8 x 8 x 256)
        add1 = up2 + up14  # add1 (8 x 8 x 256)

        # recursion (from org code): num_down_sample = 1
        low21 = self.rb13(add1)  # low21 (8 x 8 x 256)
        low31 = self.rb14(low21)  # low31 (8 x 8 x 256)
        up21 = F.interpolate(low31, scale_factor=2, mode="nearest")  # up2 (16 x 16 x 256)
        add2 = up21 + up13  # add2 (16 x 16 x 256)

        # recursion (from org code): num_down_sample = 2
        low22 = self.rb15(add2)  # low22 (16 x 16 x 256)
        low32 = self.rb16(low22)  # low32 (16 x 16 x 256)
        up22 = F.interpolate(low32, scale_factor=2, mode="nearest")  # up22 (32 x 32 x 256)
        add3 = up22 + up12  # add3 (32 x 32 x 256)

        # recursion (from org code): num_down_sample = 3
        low23 = self.rb17(add3)  # low23 (32 x 32 x 256)
        low33 = self.rb18(low23)  # low33 (32 x 32 x 256)
        up23 = F.interpolate(low33, scale_factor=2, mode="nearest")  # up23 (64 x 64 x 256)
        add4 = up23 + up11  # add4 (64 x 64 x 256)

        # recursion (from org code): num_down_sample = 4
        low24 = self.rb19(add4)  # low24 (64 x 64 x 256)
        low34 = self.rb20(low24)  # low34 (64 x 64 x 256)
        up24 = F.interpolate(low34, scale_factor=2, mode="nearest")  # up24 (128 x 128 x 256)
        add5 = up24 + up1  # add5 (128 x 128 x 256)

        return add5


class MVLMModel(nn.Module):
    def __init__(self, n_landmarks=73, n_features=256, dropout_rate=0.2, image_channels="geometry"):
        super().__init__()
        self.out_features = n_landmarks
        self.features = n_features
        self.dropout_rate = dropout_rate

        if image_channels == "geometry":
            self.in_channels = 1
        elif image_channels == "RGB":
            self.in_channels = 3
        elif image_channels == "depth":
            self.in_channels = 1
        elif image_channels == "RGB+depth":
            self.in_channels = 4
        elif image_channels == "geometry+depth":
            self.in_channels = 2
        else:
            print("Image channels should be: geometry, RGB, depth, RGB+depth or geometry+depth")
            self.in_channels = 1

        self.conv1 = nn.Conv2d(self.in_channels, int(self.features / 4), kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(int(self.features / 4))
        self.conv2 = ResidualBlock(int(self.features / 4), int(self.features / 2))
        self.conv3 = ResidualBlock(int(self.features / 2), int(self.features / 2))
        self.conv4 = ResidualBlock(int(self.features / 2), self.features)
        self.hg1 = HourGlassModule(self.features)
        self.hg2 = HourGlassModule(self.features)
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.conv5 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.features)
        self.conv6 = nn.Conv2d(self.features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(self.out_features, self.features, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.conv9 = nn.Conv2d(self.features, self.features, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.features)
        self.conv10 = nn.Conv2d(self.features, self.out_features, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(self.out_features, self.out_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)  # x: (256 x 256 x 64)
        x = self.bn1(x)
        x = F.relu(x)

        # x = F.relu(self.bn1(self.conv1(x)), True)  # x: (256 x 256 x 64)
        x = self.conv2(x)  # x: (256 x 256 x 128)
        x = F.max_pool2d(x, 2)  # x: (128 x 128 x 128)
        x = self.conv3(x)  # x: (128 x 128 x 128)
        r3 = self.conv4(x)  # r3: (128 x 128 x 256)
        x = self.hg1(r3)  # x: (128 x 128 x 256)
        x = self.dropout1(x)  # x: (128 x 128 x 256)
        ll1 = F.relu(self.bn2(self.conv5(x)), True)  # x: (128 x 128 x 256)
        x = self.conv6(ll1)  # x: (128 x 128 x NL)
        up_temp = F.interpolate(x, scale_factor=2, mode="nearest")  # up_temp (256 x 256 x NL)
        up_out = self.conv8(up_temp)  # up_out (256 x 256 x NL)
        x = self.conv7(x)  # x: (128 x 128 x 256)

        sum_temp = r3 + ll1 + x  # sum_temp: (128 x 128 x 256)

        x = self.hg2(sum_temp)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv9(x)), True)  # x: (128 x 128 x 256)
        x = self.conv10(x)  # x: (128 x 128 x NL)
        up_temp2 = F.interpolate(x, scale_factor=2, mode="nearest")  # up_temp2 (256 x 256 x NL)
        up_out2 = self.conv11(up_temp2)  # up_out2 (256 x 256 x NL)

        outputs = torch.stack([up_out, up_out2])
        return outputs
