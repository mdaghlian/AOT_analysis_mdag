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

def _cart2pol(x, y):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return r, theta

def _pol2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y    

def add_pol_cart(pdict):
    if 'x' in pdict.keys():
        ecc,pol = _cart2pol(pdict['x'], pdict['y'])
        pdict['ecc'] = ecc
        pdict['pol'] = pol
    elif 'ecc' in pdict.keys():
        x,y = _pol2cart(pdict['ecc'], pdict['pol'])
        pdict['x']=x
        pdict['y']=y
    return pdict


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
            'size': params['spatial_env']*self.dov_per_scr1,   # s.d (image) * (deg/image) = deg
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

    def filter_selectivity(self, w, param):
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

    def filter_weighted_mean(self, w, params=None, option='clamp'):
        """
        Compute weighted means for a set of weight vectors.
        weights: array of shape (n_filters, N)
        Returns: DataFrame of shape (N, len(params))
        """        
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")
        if params is None:
            params = self.params_list
        # sum of weights per column
        # How to deal with negative weights? 
        # -> problem is you can get sumw nearly 0...
        if option=='clamp-pos':
            # No negative weights
            w = np.maximum(w, 0)
            wsum = w.sum(axis=0)
        elif 'clamp-top-n-' in option:
            # Only include the top N weights...
            x = int(option.split('-')[-1])            
            abs_w = np.abs(w)
            n,m = abs_w.shape
            x = 50
            # 2) For each column j, find the row‐indices of the top x absolute values.
            #    np.argpartition(abs_w, -x, axis=0) rearranges each column so that
            #    its x largest entries (by absolute value) are in the last x rows (in arbitrary order).
            #    By slicing [-x:] along axis=0, we get those row‐indices for each column.
            top_rows = np.argpartition(abs_w, -x, axis=0)[-x:, :]   # shape: (x, m)

            # 3) Build a boolean mask of shape (n, m), initially False.
            keep_mask = np.zeros((n, m), dtype=bool)

            # 4) We need to set keep_mask[i, j] = True whenever i ∈ top_rows[:, j].
            #    Make a “column index” array of shape (x, m) so we can do this all at once:
            cols = np.arange(m)                       # shape: (m,)
            cols_broadcast = np.broadcast_to(cols, (x, m))  # shape: (x, m)

            # 5) Mark those positions as True:
            keep_mask[top_rows, cols_broadcast] = True

            # 6) Finally, zero out everything except where keep_mask is True:
            w = np.where(keep_mask, w, 0)            
            wsum = abs_w.sum(axis=0)
        elif option=='L1':
            # Avoids wsum=0
            wsum = np.abs(w).sum(axis=0)
        elif option=='none':
            wsum = w.sum(axis=0)
        
        mpos = {}
        mpos['x'] = ((w * self._arrays['x'][:, None]).sum(axis=0) / wsum)
        mpos['y'] = ((w * self._arrays['y'][:, None]).sum(axis=0) / wsum)
        mpos['ecc'],mpos['pol']  = self._cart2pol(mpos['x'], mpos['y'])

        out = {}
        for p in params: 
            if p in ['x', 'y', 'ecc', 'pol']:
                out[p] = mpos[p]
            elif p in ['SF', 'size', 'TF', 'vel']:
                # straightforward means
                arr = self._arrays[p][:, None]  # shape (F,1)
                mean_vals = (w * arr).sum(axis=0) / wsum
                out[p] = mean_vals
            elif p in ['dir']:
                # direction: circular mean
                theta = np.deg2rad(self._arrays['dir'])[:, None]
                sin_sum = (w * np.sin(theta)).sum(axis=0)
                cos_sum = (w * np.cos(theta)).sum(axis=0)
                dir_mean = (np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360)
                out['dir'] = dir_mean
        # assemble DataFrame
        return pd.DataFrame(out)

    def filter_max(self, w, params=None, option='pos'):
        """
        For each weight vector, return parameters at index of max weight.
        weights: array (n_filters, N)
        Returns: DataFrame (N, len(params))
        """
        w = np.asarray(w)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")

        if params is None:
            params = self.params_list
        # index of max per column
        if option == 'pos':
            idx = w.argmax(axis=0)
        elif option == 'abs':
            idx = np.abs(w).argmax(axis=0)
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
        return _cart2pol(x, y)
    
    def _pol2cart(self, r, theta):
        return _pol2cart(r, theta)

    def test(self):
        pprint(self.pyramid)
        pprint(self.pyramid.parameters)



class RespEstimate(FilterInfo):
    def __init__(self, **kwargs):
        # Initialize with the FilterInfo class..
        super().__init__(**kwargs)        

    def stim_make_space(self,nframe=60, grid_size=(17, 17), **kwargs):
        # Going to splid 
        print(self.vhsize)
        downsample = kwargs.get('downsample', 1)
        vhsize = self.vhsize[0]//downsample, self.vhsize[1]//downsample
        print(f'Downsampled vhsize = {vhsize}')
        print(self.vhsize)
        self.stim_space = {}
        self.stim_space['downsample'] = downsample
        self.stim_space['grid_size'] = grid_size
        self.stim_space['filt_resp'] = []
        self.stim_space['filt_resp_avg'] = []
        self.stim_space['stim_x']   = []
        self.stim_space['stim_y']   = []

        i_total = 0
        for ix in range(grid_size[0]):
            for iy in range(grid_size[1]):
                mat,pix_x,pix_y = generate_dynamic_gaussian_noise(
                    nframe=nframe, 
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
                self.stim_space['stim_x'].append(xdov)
                self.stim_space['stim_y'].append(ydov)
                fresp = self.pyramid.project_stimulus(mat,use_cuda=True,)
                self.stim_space['filt_resp'].append(fresp.copy())
                self.stim_space['filt_resp_avg'].append(
                    np.mean(fresp, axis=0)
                )
                    
    def stim_make_gray(self,nframe=60, **kwargs):
        # Going to splid 
        downsample = kwargs.get('downsample', 1)
        vhsize = self.vhsize[0]//downsample, self.vhsize[1]//downsample
        print(f'Downsampled vhsize = {vhsize}')
        self.stim_gray = {}
        self.stim_gray['downsample'] = downsample
        mat = np.zeros((nframe, vhsize[0], vhsize[1]), dtype=float) + 50.0
        fresp =self.pyramid.project_stimulus(mat,use_cuda=True,)
        self.stim_gray['filt_resp'] = fresp.copy()
        self.stim_gray['filt_resp_avg'] = np.mean(fresp, axis=0)        

    def stim_make_tfsf(self, nframe=60,  **kwargs):
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
                        nframe=nframe,
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
    
    # ******************************
    def stim_resp_space(self, w, **kwargs):
        ''' Compute spatial preference
        
        w : weights n filters x n vox

        Return 

        mat : X x Y x n vox 
        -> where X and Y are taken from the 
        '''
        return_idx = kwargs.get('return_idx', False)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")                

        gray_resp = self.stim_gray_resp(w)
        filt_resp_mat = np.vstack(self.stim_space['filt_resp_avg'])
        space_resp = (filt_resp_mat @ w) - gray_resp[...,np.newaxis].T
        # Now average over 
        xs = np.array(self.stim_space['stim_x'])
        u_xs = np.unique(xs)
        ys = np.array(self.stim_space['stim_y'])
        u_ys = np.unique(ys)
        mat = np.zeros((len(u_xs), len(u_ys), w.shape[1]))
        idx_mat = {
            'x' : np.zeros((len(u_xs), len(u_ys))),
            'y' : np.zeros((len(u_xs), len(u_ys))),
        }
        for i,x in enumerate(u_xs):
            for j,y in enumerate(u_ys):
                match = (xs==x) & (ys==y)
                mat[i,j,:] = np.mean(space_resp[match,:], axis=0)
                idx_mat['x'][i,j] = x
                idx_mat['y'][i,j] = y
        if return_idx:
            return mat, idx_mat
        return mat        
    
    def stim_pref_space(self, w, **kwargs):
        '''Return the preference
        '''
        wspace = self.stim_resp_space(w=w, return_idx=True, **kwargs)
        return self._stim_pref_space(wspace=wspace)        

    def _stim_pref_space(self, wspace):
        # Get indices of maximum value for each slice
        max_xy = [np.argmax(wspace[:,:,i].flatten()) for i in range(wspace.shape[-1])] 
        # Now the 
        # max_xy will contain the (x,y) coordinates for the maximum value in each slice
        # max_xy[:, 0] contains all x coordinates 
        # max_xy[:, 1] contains all y coordinates
        max_x = np.array([self.stim_space['stim_x'][i] for i in max_xy])
        max_y = np.array([self.stim_space['stim_y'][i] for i in max_xy])        
        stim_pref = {
            'x' : max_x,
            'y' : max_y,
        }
        stim_pref = add_pol_cart(stim_pref)
        return stim_pref

    def stim_resp_tfsf(self, w, **kwargs):
        ''' Compute tf & sf preference
        '''
        return_idx = kwargs.get('return_idx', False)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")                

        gray_resp = self.stim_gray_resp(w)
        filt_resp_mat = np.vstack(self.stim_tfsf['filt_resp_avg'])
        tfsf_resp = (filt_resp_mat @ w) - gray_resp[...,np.newaxis].T

        # Now average over 
        tfs = np.array(self.stim_tfsf['tf'])
        u_tfs = np.unique(tfs)
        sfs = np.array(self.stim_tfsf['sf'])
        u_sfs = np.unique(sfs)
        mat = np.zeros((len(u_sfs), len(u_tfs), w.shape[1]))
        idx_mat = {
            'tf' : np.zeros((len(u_sfs), len(u_tfs))),
            'sf' : np.zeros((len(u_sfs), len(u_tfs)))
        }
        for i,sf in enumerate(u_sfs):
            for j,tf in enumerate(u_tfs):
                match = (sfs==sf) & (tfs==tf)
                mat[i,j,:] = np.mean(tfsf_resp[match,:], axis=0)
                idx_mat['sf'][i,j]=sf
                idx_mat['tf'][i,j]=tf
        if return_idx:
            return mat, idx_mat
        return mat    

    def stim_pref_tfsf(self, w, **kwargs):
        '''Return the preference
        '''
        wtfsf,idx = self.stim_resp_tfsf(w=w,  return_idx=True,**kwargs)
        return self._stim_pref_tfsf(wtfsf=wtfsf, idx=idx)        

    def _stim_pref_tfsf(self, wtfsf,idx):
        # Get indices of maximum value for each slice
        max_tfsf = [np.argmax(wtfsf[:,:,i].flatten()) for i in range(wtfsf.shape[-1])] 
        # Now the 
        idx_tf = idx['tf'].flatten()
        idx_sf = idx['sf'].flatten()
        max_tf = np.array([idx_tf[i] for i in max_tfsf])
        max_sf = np.array([idx_sf[i] for i in max_tfsf])        
        vel = np.zeros_like(max_sf)
        valid_vel = (max_sf!=0) & (max_tf!=0)
        vel[valid_vel] = max_tf[valid_vel] / max_sf[valid_vel]
        stim_pref = {
            'tf' : max_tf,
            'sf' : max_sf,
            'vel' : vel
        }
        return stim_pref

    def stim_resp_dir(self, w, **kwargs):
        ''' Compute tf & sf preference
        '''
        return_idx = kwargs.get('return_idx', False)
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[0] != self.n_filters:
            raise ValueError("Weights must have shape (n_filters, N)")                

        gray_resp = self.stim_gray_resp(w)
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
        idx_mat = {
            'dir' : np.zeros(len(u_dirs)),
        }
        for i,dir in enumerate(u_dirs):
            # Only where SF & TF is not zero 
            match = (sfs!=0) & (tfs!=0) & (dirs==dir)
            mat[i,:] = np.mean(tfsf_resp[match,:], axis=0)
            idx_mat['dir'][i] = dir
        if return_idx:
            return mat, idx_mat
        return mat
    
    def stim_pref_dir(self, w, **kwargs):
        '''Return the preference
        '''
        wdir,idx = self.stim_resp_dir(w=w,  return_idx=True,**kwargs)
        return self._stim_pref_dir(wdir=wdir, idx=idx)        

    def _stim_pref_dir(self, wdir, idx):
        # Get indices of maximum value for each slice
        max_dir_idx = [np.argmax(wdir[:,i].flatten()) for i in range(wdir.shape[-1])] 
        idx_dir = idx['dir'].flatten()
        # Now the 
        max_dir = np.array([idx_dir[i] for i in max_dir_idx])
        stim_pref = {
            'dir' : max_dir,
        }
        return stim_pref

    def stim_gray_resp(self, w):
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

    def nish_stim_plot(self, w, **kwargs):
        return_fig = kwargs.get('return_fig', False)
        import matplotlib.pyplot as plt
        tfsf, idx_tfsf = self.stim_resp_tfsf(w, return_idx=True)
        space, idx_space = self.stim_resp_space(w, return_idx=True)
        fig, ax = plt.subplots(1,2, figsize=(10,5), width_ratios=(4,1))
        vlim_tfsf = np.max(np.abs(tfsf))
        vlim_space = np.max(np.max(np.abs(space)))

        xs = np.array(self.stim_space['stim_x'])
        u_xs = np.unique(xs)
        ys = np.array(self.stim_space['stim_y'])
        u_ys = np.unique(ys)        
        # Define image extent
        # extent = [-1*self.aspect_ratio, 1*self.aspect_ratio, -1, 1]
        extent = [-self.screen_width_dov/2,self.screen_width_dov/2,-self.screen_height_dov/2, self.screen_height_dov/2]

        # Compute tick positions to center the labels in each grid cell
        x_ticks = np.linspace(extent[0], extent[1], len(u_xs), endpoint=False) + (extent[1] - extent[0]) / len(u_xs) / 2
        y_ticks = np.linspace(extent[2], extent[3], len(u_ys), endpoint=False) + (extent[3] - extent[2]) / len(u_ys) / 2

        img_space = ax[0].imshow(
            # np.flipud(idx_space['x'].T), #space, 
            np.flipud(space.squeeze().T),
            vmin=-vlim_space,vmax=vlim_space,
            cmap='RdBu_r',
            extent=extent
        )
        plt.colorbar(img_space, ax=ax[0])

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
        img_tfsf = ax[1].imshow(
            tfsf, 
            vmin=-vlim_tfsf, vmax=vlim_tfsf, 
            cmap='RdBu_r',
            extent=extent, origin='lower'  # origin='lower' ensures y-ticks align properly
        )
        

        # Set centered ticks and labels
        ax[1].set_xticks(x_ticks)
        ax[1].set_xticklabels([f'{i:.2f}' for i in u_tfs])
        ax[1].set_yticks(y_ticks)
        ax[1].set_yticklabels([f'{i:.2f}' for i in u_sfs])
        ax[1].set_xlabel('TF')
        ax[1].set_ylabel('SF')
        plt.colorbar(img_tfsf, ax=ax[1])
        if return_fig:
            return fig, ax

    def nish_stim_plot2(self, w, **kwargs):
        return_fig = kwargs.get('return_fig', False)
        import matplotlib.pyplot as plt
        tfsf, idx_tfsf = self.stim_resp_tfsf(w, return_idx=True)
        space, idx_space = self.stim_resp_space(w, return_idx=True)
        fig, ax = plt.subplots(1,2, figsize=(10,5), width_ratios=(4,1))
        vlim_tfsf = np.max(np.abs(tfsf))
        vlim_space = np.max(np.max(np.abs(space)))
        print(space.shape)
        print(idx_space['x'].shape)
        pcm = ax[0].scatter(
            x=idx_space['x'].flatten(), y=idx_space['y'].flatten(),
            c=space.flatten(), 
            # shading='auto', 
            vmin=-vlim_space,vmax=vlim_space,cmap='RdBu_r',
        )



def generate_dynamic_gaussian_noise(grid_size, grid_location, vhsize, nframe, mean=50.0, low=0.0, high=100.0, seed=None, **kwargs):
    """
    Generates a dynamic Gaussian white noise movie for a specific grid cell.

    Returns:
    --------
    movie : np.ndarray, shape (nframe, cell_height, cell_width)
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
    movie_full = np.zeros((nframe, pix_y, pix_x), dtype=float) + mean

    # Generate Gaussian white noise for the patch region
    noise_patch = np.random.uniform(low=low, high=high, size=(nframe, cell_height, cell_width))

    # Insert noise into the full-screen movie
    movie_full[:, y_start:y_end, x_start:x_end] = noise_patch

    return movie_full, mid_x, mid_y

def mk_drifting_grating_movie(vhsize,
                               stimulus_fps,
                               aspect_ratio='auto',
                               nframe=60,
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


########

import numpy as np
import numpy as np


def fit_sym_gaussians_with_offset_moments(X, Y, Z):
    """
    Fit a symmetric 2D Gaussian + constant offset to each slice Z[:,:,k]
    using analytic (moment‐based) estimates, and compute R² for each fit.

    Parameters
    ----------
    X : ndarray, shape (n, m)
        x-coordinate at each pixel.
    Y : ndarray, shape (n, m)
        y-coordinate at each pixel.
    Z : ndarray, shape (n, m, c)
        Measured intensities.

    Returns
    -------
    pdict : dict with keys
        'x'         : ndarray, shape (c,)
                      Fitted x0_k for each slice k.
        'y'         : ndarray, shape (c,)
                      Fitted y0_k for each slice k.
        'size'      : ndarray, shape (c,)
                      Fitted σ_k for each slice k.
        'amplitude' : ndarray, shape (c,)
                      Fitted A_k for each slice k.
        'offset'    : ndarray, shape (c,)
                      Fitted background offset b_k for each slice k.
        'r2'        : ndarray, shape (c,)
                      R² of each fit (coefficient of determination). NaN if degenerate.
    """
    # Number of slices
    _, _, num_slices = Z.shape

    # Pre-allocate outputs
    xs = np.zeros(num_slices, dtype=float)
    ys = np.zeros(num_slices, dtype=float)
    sigmas = np.zeros(num_slices, dtype=float)
    amps = np.zeros(num_slices, dtype=float)
    offs = np.zeros(num_slices, dtype=float)
    r2s = np.zeros(num_slices, dtype=float)

    # Flattened coordinates (same for every slice)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    coords = np.vstack((X_flat, Y_flat))  # shape (2, n*m)

    for k in range(num_slices):
        Zk = Z[:, :, k]
        Z_flat = Zk.ravel()

        # 1) Estimate offset b as the minimum pixel value
        b_hat = np.min(Z_flat)
        offs[k] = b_hat

        # 2) Subtract offset and clip negative residuals to zero
        Zp = Z_flat - b_hat
        Zp[Zp < 0] = 0.0

        # 3) Total “mass”
        M = np.sum(Zp)
        if M <= 0:
            # Degenerate slice (flat or all background)
            xs[k] = 0.0 #np.nan
            ys[k] = 0.0 #np.nan
            sigmas[k] =0.0 # np.nan
            amps[k] = 0.0 #np.nan
            r2s[k] = 0.0 #np.nan
            continue

        # 4) Centroid (first moments)
        x0 = np.sum(X_flat * Zp) / M
        y0 = np.sum(Y_flat * Zp) / M
        xs[k], ys[k] = x0, y0

        # 5) Second moment for sigma
        dx2 = (X_flat - x0)**2 + (Y_flat - y0)**2
        S = np.sum(dx2 * Zp)
        sigma2 = S / M
        sigma = np.sqrt(sigma2)
        sigmas[k] = sigma

        # 6) Amplitude: peak minus background
        A = np.max(Zp)
        amps[k] = A

        # 7) Compute R²
        #    Z_pred(i,j) = A * exp( - ((x_i-x0)^2+(y_j-y0)^2) / (2σ²) ) + b_hat
        exponent = -dx2 / (2 * sigma2)
        Z_pred_flat = A * np.exp(exponent) + b_hat

        #    SS_res and SS_tot
        ss_res = np.sum((Z_flat - Z_pred_flat)**2)
        z_mean = np.mean(Z_flat)
        ss_tot = np.sum((Z_flat - z_mean)**2)

        if ss_tot == 0:
            # If all Z_flat are identical, define R² = 1 if model matches exactly
            if np.allclose(Z_flat, Z_pred_flat, atol=1e-8):
                r2 = 1.0
            else:
                r2 = 0.0
        else:
            r2 = 1.0 - (ss_res / ss_tot)
        r2s[k] = r2

    pdict = {
        'x': xs,
        'y': ys,
        'size': sigmas,
        'amplitude': amps,
        'offset': offs,
        'r2': r2s,
    }
    pdict = add_pol_cart(pdict)

    return pdict
