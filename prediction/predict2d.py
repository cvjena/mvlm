import imageio
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
import random
import math


class Predict2D:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device

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
        im_size = image.size(2)
        hm_size = heat_map.shape[2]

        # Super hacky way to convert gray to RGB
        # show first image in batch
        # TODO make it accept both grey and color images
        i = np.zeros((im_size, im_size, 3))
        i[:, :, 0] = image[0, :, :]
        i[:, :, 1] = image[0, :, :]
        i[:, :, 2] = image[0, :, :]

        # Generate combined heatmap image in RGB channels.
        # This must be possible to do smarter - Alas! My Python skillz are lacking
        hm = np.zeros((hm_size, hm_size, 3))
        n_lm = heat_map.shape[0]
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

        im_marked = self.generate_image_with_heatmap_maxima(i, heat_map)

        plt.figure()
        plt.imshow(i)
        plt.figure()
        plt.imshow(hm)
        plt.figure()
        plt.imshow(im_marked)
        plt.axis('off')
        plt.ioff()
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

    def predict_heatmaps_from_images(self, image_stack):
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
        print("Prediction [0] - GPU time: ", time.time() - t)
        print("Prediction [0] - GPU time (mean): ", np.mean(pre_times))
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t = time.time()        
        heatmaps = heatmaps.cpu()
        print("Prediction [1] - Copy to CPU: ", time.time() - t)
        
        t = time.time()
        self.find_maxima_in_batch_of_heatmaps(heatmaps,  heatmap_maxima)
        print("Prediction [2] - Find maxima: ", time.time() - t)
        return heatmap_maxima
