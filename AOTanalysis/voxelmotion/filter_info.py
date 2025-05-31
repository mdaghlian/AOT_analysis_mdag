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



class NishiStim(FilterInfo):
    def __init__(self, **kwargs):
        # Initialize with the FilterInfo class..
        super().__init__(**kwargs)
        

    def _stim_make_space(self,n_frames=24, grid_size=(17, 17), **kwargs):
        # Going to splid 
        downsample = kwargs.get('downsample', 1)
        vhsize = self.vhsize[0]//downsample, self.vhsize[1]//downsample
        print(f'Downsampled vhsize = {vhsize}')
        self.stim_space = {}
        self.stim_space['downsample'] = downsample
        self.stim_space['grid_size'] = grid_size
        self.stim_space['filt_resp'] = []
        self.stim_space['filt_resp_avg'] = []
        self.stim_space['filt_x']   = []
        self.stim_space['filt_y']   = []

        i_total = 0
        for ix in range(grid_size[0]):
            for iy in range(grid_size[1]):
                mat,pix_x,pix_y = generate_dynamic_gaussian_noise(
                    n_frames=n_frames, 
                    grid_size=grid_size, 
                    vhsize=vhsize, 
                    grid_location=[ix,iy], 
                    **kwargs,
                    )
                # convert pix_x, pix_y to fractions of height
                xdov, ydov = self._screen2dov(
                    xscr=pix_x / vhsize[0], # Divide be new screen heigh in pix
                    yscr=pix_y / vhsize[0],

                )
                self.stim_space['filt_x'].append(xdov)
                self.stim_space['filt_y'].append(ydov)
                fresp = self.pyramid.project_stimulus(mat,use_cuda=True,)
                self.stim_space['filt_resp'].append(fresp.copy())
                self.stim_space['filt_resp_avg'].append(
                    np.mean(fresp, axis=0)
                )
                
    
    def _stim_make_gray(self,n_frames=24, **kwargs):
        # Going to splid 
        downsample = kwargs.get('downsample', 1)
        vhsize = self.vhsize[0]//downsample, self.vhsize[1]//downsample
        print(f'Downsampled vhsize = {vhsize}')
        self.stim_gray = {}
        self.stim_gray['downsample'] = downsample
        mat = np.zeros((n_frames, vhsize[0], vhsize[1]), dtype=float) + 50.0
        fresp =self.pyramid.project_stimulus(mat,use_cuda=True,)
        self.stim_gray['filt_resp'] = fresp.copy()
        self.stim_gray['filt_resp_avg'] = np.mean(fresp, axis=0)        

    def _stim_make_tfsf(self, n_frames=24,  **kwargs):
        '''
        Generate full-field drifting sine-wave gratings at specified temporal
        frequency (tf), spatial frequency (sf), and orientation (dir).
        '''
        downsample = kwargs.get('downsample', 1)
        vhsize = self.vhsize[0]//downsample, self.vhsize[1]//downsample
        tf_list = kwargs.get('tf_list', self.pyramid.definition['temporal_frequencies'])
        sf_list = kwargs.get('sf_list', self.pyramid.definition['spatial_frequencies'])
        dir_list = kwargs.get('dir_list', self.pyramid.definition['spatial_directions'])
        frame_rate = kwargs.get('frame_rate', self.pyramid.definition['stimulus_fps'])
        
        print(f'Downsampled vhsize = {vhsize}')
        self.stim_tfsf = {}
        self.stim_tfsf['downsample'] = downsample
        self.stim_tfsf['tf'] = []
        self.stim_tfsf['sf'] = []
        self.stim_tfsf['dir'] = []
        self.stim_tfsf['filt_resp'] = []
        self.stim_tfsf['filt_resp_avg'] = []

        for tf in tf_list:
            for sf in sf_list:
                for dir in dir_list:
                    print(f'{tf}, {sf}, {dir}')
                    # Generate drifting grating movie
                    mat = mk_drifting_grating_movie(
                        vhsize=vhsize,
                        stimulus_fps=frame_rate,
                        aspect_ratio=self.aspect_ratio,
                        nframe=n_frames,
                        direction=dir,
                        spatial_freq=sf,
                        temporal_freq=tf,
                    )
                    self.stim_tfsf['tf'].append(tf)
                    self.stim_tfsf['sf'].append(sf)
                    self.stim_tfsf['dir'].append(dir)
                    fresp =self.pyramid.project_stimulus(mat,use_cuda=True,)
                    self.stim_tfsf['filt_resp'].append(fresp.copy())
                    self.stim_tfsf['filt_resp_avg'].append(np.mean(fresp, axis=0))        
                    if sf==0.0:
                        # only one direction for sf 0
                        break
    def _pref_space(self, w):
        ''' Compute spatial preference
        
        w : weights n filters x n vox

        Return 

        mat : X x Y x n vox 
        -> where X and Y are taken from the 
        '''
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")                

        gray_resp = self._gray_resp(w)
        filt_resp_mat = np.vstack(self.stim_space['filt_resp_avg'])
        space_resp = (filt_resp_mat @ w) - gray_resp[...,np.newaxis].T
        # Now average over 
        xs = np.array(self.stim_space['filt_x'])
        u_xs = np.unique(xs)
        ys = np.array(self.stim_space['filt_y'])
        u_ys = np.unique(ys)
        mat = np.zeros((len(u_xs), len(u_ys), w.shape[1]))
        for i,x in enumerate(u_xs):
            for j,y in enumerate(u_ys):
                match = (xs==x) & (ys==y)
                mat[i,j,:] = np.mean(space_resp[match,:], axis=0)
        return mat        
    
    def _pref_tfsf(self, w):
        ''' Compute tf & sf preference
        '''
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")                

        gray_resp = self._gray_resp(w)
        filt_resp_mat = np.vstack(self.stim_tfsf['filt_resp_avg'])
        tfsf_resp = (filt_resp_mat @ w) - gray_resp[...,np.newaxis].T

        # Now average over 
        tfs = np.array(self.stim_tfsf['tf'])
        u_tfs = np.unique(tfs)
        sfs = np.array(self.stim_tfsf['sf'])
        u_sfs = np.unique(sfs)
        mat = np.zeros((len(u_sfs), len(u_tfs), w.shape[1]))
        print(mat.shape)
        for i,sf in enumerate(u_sfs):
            for j,tf in enumerate(u_tfs):
                match = (sfs==sf) & (tfs==tf)
                mat[i,j,:] = np.mean(tfsf_resp[match,:], axis=0)

        return mat
    

    def _pref_dir(self, w):
        ''' Compute tf & sf preference
        '''
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")                

        gray_resp = self._gray_resp(w)
        filt_resp_mat = np.vstack(self.stim_tfsf['filt_resp_avg'])
        tfsf_resp = (filt_resp_mat @ w) - gray_resp[...,np.newaxis].T

        # Now average over 
        tfs = np.array(self.stim_tfsf['tf'])
        u_tfs = np.unique(tfs)        
        sfs = np.array(self.stim_tfsf['sf'])
        u_sfs = np.unique(sfs)
        dirs = np.array(self.stim_tfsf['dir'])
        u_dirs = np.unique(dirs)

        mat = np.zeros((len(u_dirs), w.shape[1]))
        print(mat.shape)
        for i,dir in enumerate(u_dirs):
            
            match = (sfs!=0) & (tfs!=0) & (dirs==dir)
            mat[i,:] = np.mean(tfsf_resp[match,:], axis=0)

        return mat


    def _gray_resp(self, w):
        ''' Compute weighted response to gray stimuli

        w : n filter x n vox

        Return: gray_resp : n vox
        '''
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")        

        gray_resp = (self.stim_gray['filt_resp_avg'][...,np.newaxis] * w).sum(axis=0)
        return gray_resp

    def nish_stim_plot(self, w):
        import matplotlib.pyplot as plt
        tfsf = self._pref_tfsf(w)
        space = self._pref_space(w)
        fig, ax = plt.subplots(1,2, figsize=(10,5), width_ratios=(4,1))
        vlim = np.max(np.abs(tfsf))
        vlim = np.max([np.max(np.abs(space)), vlim])

        xs = np.array(self.stim_space['filt_x'])
        u_xs = np.unique(xs)
        ys = np.array(self.stim_space['filt_y'])
        u_ys = np.unique(ys)        
        # Define image extent
        extent = [-1*self.aspect_ratio, 1*self.aspect_ratio, -1, 1]

        # Compute tick positions to center the labels in each grid cell
        x_ticks = np.linspace(extent[0], extent[1], len(u_xs), endpoint=False) + (extent[1] - extent[0]) / len(u_xs) / 2
        y_ticks = np.linspace(extent[2], extent[3], len(u_ys), endpoint=False) + (extent[3] - extent[2]) / len(u_ys) / 2

        ax[0].imshow(
            space, 
            vmin=-vlim,vmax=vlim,cmap='RdBu_r',
            extent=extent
        )

        # Set centered ticks and labels
        ax[0].set_xticks(x_ticks)
        ax[0].set_xticklabels([f'{i:.2f}' for i in u_xs])
        ax[0].set_yticks(y_ticks)
        ax[0].set_yticklabels([f'{i:.2f}' for i in u_ys])


        tfs = np.array(self.stim_tfsf['tf'])
        u_tfs = np.unique(tfs)
        sfs = np.array(self.stim_tfsf['sf'])
        u_sfs = np.unique(sfs)
        ar = len(u_tfs)/len(u_sfs)
        # Define image extent
        extent = [-1*ar, 1*ar, -1, 1]
        # Compute tick positions to center the labels in each grid cell
        x_ticks = np.linspace(extent[0], extent[1], len(u_tfs), endpoint=False) + (extent[1] - extent[0]) / len(u_tfs) / 2
        y_ticks = np.linspace(extent[2], extent[3], len(u_sfs), endpoint=False) + (extent[3] - extent[2]) / len(u_sfs) / 2

        # Plot the image
        img = ax[1].imshow(
            tfsf, 
            vmin=-vlim, vmax=vlim, cmap='RdBu_r',
            extent=extent, origin='lower'  # origin='lower' ensures y-ticks align properly
        )

        # Set centered ticks and labels
        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels([f'{i:.2f}' for i in u_tfs])
        ax[1].set_yticks(y_ticks)
        ax[1].set_yticklabels([f'{i:.2f}' for i in u_sfs])
        ax[1].set_xlabel('TF')
        ax[1].set_ylabel('SF')
        plt.colorbar(img, ax=ax[0])


def generate_dynamic_gaussian_noise(grid_size, grid_location, vhsize, n_frames, mean=50.0, low=0.0, high=100.0, seed=None, **kwargs):
    """
    Generates a dynamic Gaussian white noise movie for a specific grid cell.

    Returns:
    --------
    movie : np.ndarray, shape (n_frames, cell_height, cell_width)
        Dynamic Gaussian white noise movie for the specified grid cell.
    mid_x : int
        X coordinate (pixel) of the center of the grid cell on the full screen.
    mid_y : int
        Y coordinate (pixel) of the center of the grid cell on the full screen.
    """
    if seed is not None:
        np.random.seed(seed)

    n_x, n_y = grid_size

    # I KNOW THIS LOOKS THE WRONG WAY AROUND...
    # SO ANNOYING - but stuff is done vertical - horizontal ...
    pix_y, pix_x = vhsize
    idx_x, idx_y = grid_location

    # Compute size of each grid cell
    cell_width = pix_x // n_x
    cell_height = pix_y // n_y

    # Compute pixel range for the target cell
    x_start = idx_x * cell_width
    x_end = x_start + cell_width
    y_start = idx_y * cell_height
    y_end = y_start + cell_height

    # Center pixel coordinates of the cell on the full screen
    mid_x = x_start + cell_width / 2
    mid_y = y_start + cell_height / 2

    # Create full-screen movie initialized to zero
    movie_full = np.zeros((n_frames, pix_y, pix_x), dtype=float) + mean

    # Generate Gaussian white noise for the patch region
    noise_patch = np.random.uniform(low=low, high=high, size=(n_frames, cell_height, cell_width))

    # Insert noise into the full-screen movie
    movie_full[:, y_start:y_end, x_start:x_end] = noise_patch

    return movie_full, mid_x, mid_y

def mk_drifting_grating_movie(vhsize,
                               stimulus_fps,
                               aspect_ratio='auto',
                               nframe=24,
                               centerh=0.5,
                               direction=45.0,
                               spatial_freq=16.0,
                               temporal_freq=2.0,
                               centerv=0.5,
                               spatial_phase_offset=0.0,
                               mean=50.0, contrast=50.0,
                               **kwargs
                               ):
    '''Make a full-field drifting sinusoidal grating movie.
    
    Adapted from     
    https://github.com/gallantlab/pymoten/blob/7570e1979744fd0950e21e581dd12e07a8cd6b1b/moten/core.py#L132

    Parameters
    ----------
    vhsize : tuple of ints, (vdim, hdim)
        Frame size in pixels: (vertical, horizontal).
    stimulus_fps : float
        Playback speed (frames per second).
    aspect_ratio : 'auto' or float
        Frame width/height ratio. 'auto' sets it to hdim/vdim.
    nframe : int or 'auto'
        Number of frames. 'auto' uses one temporal period (fps/temporal_freq).
    centerh, centerv : floats in [0,1]
        Grating center in normalized coordinates (horizontal, vertical).
    direction : float [degrees]
        Drift direction (0=right, 90=up).
    spatial_freq : float
        Spatial frequency (cycles per frame width).
    temporal_freq : float
        Temporal frequency (cycles per second).
    spatial_phase_offset : float [radians]
        Initial spatial phase offset.

    Returns
    -------
    movie : ndarray, shape (nframe, vdim, hdim)
        Drifting grating frames, values in [-1,1].
    '''
    vdim, hdim = vhsize
    # aspect ratio
    if aspect_ratio == 'auto':
        aspect_ratio = hdim / float(vdim)

    nframe = int(nframe)

    # spatial coords
    dh = np.linspace(0, aspect_ratio, hdim, endpoint=True)
    dv = np.linspace(0, 1, vdim, endpoint=True)
    ihs, ivs = np.meshgrid(dh, dv)

    dt = np.arange(nframe) / stimulus_fps #np.linspace(0, 1, nframe, endpoint=False)

    # compute spatial and temporal waves
    theta = np.deg2rad(direction)
    fh = -spatial_freq * np.cos(theta) * 2 * np.pi
    fv = spatial_freq * np.sin(theta) * 2 * np.pi
    if temporal_freq == 0:
        ft = np.ones_like(temporal_freq)
    else:
        ft = temporal_freq * (1.0 / temporal_freq) * 2 * np.pi  # one cycle over dt


    spatial_arg = (ihs - centerh) * fh + (ivs - centerv) * fv + spatial_phase_offset

    # build movie
    movie = np.zeros((nframe, vdim, hdim), dtype=np.float32)
    for t in range(nframe):
        temporal_phase = dt[t] * ft
        movie[t] = mean + contrast * np.sin(spatial_arg + temporal_phase)

    return movie






if __name__ == "__main__":
    # Create a filter info object
    filter_info = NishiStim()
    # Test the filter info class
    filter_info.test()
    # Print the filter info class
    pprint(filter_info.__dict__)