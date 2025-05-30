import moten
import os
import numpy as np
from pathlib import Path
from pprint import pprint
import pandas as pd
# import aot


# sample_video = "/tank/shared/2024/visual/AOT/derivatives/stimuli/rescaled_final/0001_fw.mp4"
# spatial_frequencies = [0, 2, 4, 8, 16, 32]
# # Create a pyramid of spatio-temporal gabor filters
# luminance_images = moten.io.video2luminance(sample_video)
# nimages, vdim, hdim = luminance_images.shape
# print(nimages, vdim, hdim)
# # 60 1080 1920

# pyramid = moten.get_default_pyramid(
#     vhsize=(vdim, hdim), fps=24, spatial_frequencies=spatial_frequencies
# )
# # 11845 filters





class FilterInfo:
    def __init__(self, vhsize=(1080, 1920), fps=24,
                 spatial_frequencies=[0, 2, 4, 8, 16, 32], **kwargs):
        self.vhsize = vhsize
        self.screen_height_pix = self.vhsize[0]
        self.screen_width_pix = self.vhsize[1]
        self.aspect_ratio = vhsize[1] / vhsize[0]
        self.fps = fps
        self.spatial_frequencies = spatial_frequencies
        pyramid = kwargs.get('pyramid', None)
        if pyramid is None:
            self.pyramid = moten.get_default_pyramid(
                vhsize=self.vhsize, fps=self.fps,
                spatial_frequencies=self.spatial_frequencies
            )
        else:
            self.pyramid = pyramid 
        # screen geometry
        self.screen_distance = kwargs.get("screen_distance", 196)
        self.screen_height = kwargs.get("screen_height", 39.3)
        self.screen_width = self.aspect_ratio * self.screen_height
        self.screen_height_dov = 2 * np.degrees(
            np.arctan(self.screen_height / (2 * self.screen_distance))
        )
        self.screen_width_dov = 2 * np.degrees(
            np.arctan(self.screen_width / (2 * self.screen_distance))
        )

        # 1 in screen coordinates corresponds dov_per_image in DoV
        self.dov_per_scr1 = self.screen_height_dov  

        # build filter dataframe
        params = self.pyramid.parameters
        # center in dov
        x_dov, y_dov = self._screen2dov(params['centerh'], params['centerv'])
        ecc, pol = self._cart2pol(x_dov, y_dov)

        data = {            
            'x': x_dov,
            'y': y_dov,
            'ecc': ecc,
            'pol': pol,
            'SFimg': params['spatial_freq'],
            'SF': params['spatial_freq']/self.dov_per_scr1,  # cycle/image * 1/(deg/image) = cycle/degree 
            'size': params['spatial_env']/self.dov_per_scr1,   # s.d (image) * 1/(deg/image) = deg
            'TF': params['temporal_freq'],
            'dir': params['direction'],
        }
        # velocity
        vel = data['TF'] / data['SF']
        vel[(data['SF'] == 0) | (data['TF'] == 0)] = 0
        data['vel'] = vel

        self.filter_df = pd.DataFrame(data)
        self.params_list = list(self.filter_df.columns)
        self.n_filters = len(self.filter_df)

        # cache numpy arrays for speed
        self._arrays = {p: self.filter_df[p].to_numpy() for p in self.params_list}

    def p_selectivity(self, w, param):
        """ compute the selectivity to a features
        selectivity_index = (Max Weight - min Weight) / (sum Weights)
        """
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")        
        unique = np.unique(self.filter_df[param])
        collapsed_ws = []
        for u in unique:
            idx = np.where(self.filter_df[param]==u)
            collapsed_ws.append(
                np.abs(w[idx,:].sum(axis=1))
            )
        collapsed_ws = np.vstack(collapsed_ws) # n unique x n weights
        sensitivity = (np.max(collapsed_ws, axis=0) - np.min(collapsed_ws, axis=0)) / np.sum(collapsed_ws, axis=0)
        return sensitivity

    def filter_weighted_mean(self, weights, params=None):
        """
        Compute weighted means for a set of weight vectors.
        weights: array of shape (n_filters, N)
        Returns: DataFrame of shape (N, len(params))
        """
        w = np.asarray(weights)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")

        if params is None:
            params = self.params_list
        # sum of weights per column
        wsum = w.sum(axis=0)
        out = {}
        for p in params: 
            if p in ['x', 'y', 'ecc', 'SF', 'size', 'TF', 'vel']:
                # straightforward means
                arr = self._arrays[p][:, None]  # shape (F,1)
                mean_vals = (w * arr).sum(axis=0) / wsum
                out[p] = mean_vals
            elif p in ['pol']:
                # polarity: from x,y means
                mx = ((w * self._arrays['x'][:, None]).sum(axis=0) / wsum)
                my = ((w * self._arrays['y'][:, None]).sum(axis=0) / wsum)
                _, pol_mean = self._cart2pol(mx, my)
                out['pol'] = pol_mean
            elif p in ['dir']:
                # direction: circular mean
                theta = np.deg2rad(self._arrays['dir'])[:, None]
                sin_sum = (w * np.sin(theta)).sum(axis=0)
                cos_sum = (w * np.cos(theta)).sum(axis=0)
                dir_mean = (np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360)
                out['dir'] = dir_mean
        # assemble DataFrame
        return pd.DataFrame(out)

    def filter_max(self, weights, params=None):
        """
        For each weight vector, return parameters at index of max weight.
        weights: array (n_filters, N)
        Returns: DataFrame (N, len(params))
        """
        w = np.asarray(weights)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")

        if params is None:
            params = self.params_list
        # index of max per column
        idx = w.argmax(axis=0)
        out = {}
        for p in params:
            arr = self._arrays[p]
            out[p] = arr[idx]
        return pd.DataFrame(out)


    def _screen2dov(self, xscr, yscr):
        """
        xscr from pymoten is in range [0, aspect ratio] and yscr in [0, 1].
        Convert these to degrees of visual angle (DoV) coordinates.
        """
        # Centre the screen coordinates 0 = middle of the screen
        x_c = xscr - self.aspect_ratio / 2
        y_c = yscr - 0.5
        
        # Now range is [-aspect_ratio/2, aspect_ratio/2] for x_c and [-0.5, 0.5] for y_c
        # Convert to DoV coordinates by 
        xdov = x_c * self.dov_per_scr1
        ydov = y_c * self.dov_per_scr1
        
        return xdov, ydov

    def _dov2screen(self, xdov, ydov):
        # Convert degrees of visual angle (DoV) coordinates back to screen coordinates.
        x_off = xdov / self.dov_per_scr1
        y_off = ydov / self.dov_per_scr1

        # Now range is [-aspect_ratio/2, aspect_ratio/2] for x_c and [-0.5, 0.5] for y_c
        xscr = x_off + self.aspect_ratio / 2
        yscr = y_off + 0.5
        
        return xscr, yscr

    def _cart2pol(self, x, y):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return r, theta
    
    def _pol2cart(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y    

    def test(self):
        pprint(self.pyramid)
        pprint(self.pyramid.parameters)



if __name__ == "__main__":
    # Create a filter info object
    filter_info = FilterObj()
    # Test the filter info class
    filter_info.test()
    # Print the filter info class
    pprint(filter_info.__dict__)