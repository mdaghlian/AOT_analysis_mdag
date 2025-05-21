import moten
import os
import numpy as np
from pathlib import Path
from pprint import pprint
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
    def __init__(self, vhsize=(1080, 1920), fps=24, spatial_frequencies=[0, 2, 4, 8, 16, 32]):
        self.vhsize = vhsize
        self.fps = fps
        self.spatial_frequencies = spatial_frequencies
        self.pyramid = moten.get_default_pyramid(
            vhsize=self.vhsize, fps=self.fps, spatial_frequencies=self.spatial_frequencies
        )

        
    def index_to_polar_angle(self, index):
        """
        Convert a filter index to polar angle.
        """
        aspect_ratio = self.pyramid.parameters["aspect_ratio"][0]
        centerpos = (0.5, aspect_ratio/2) # (v,h)
        centerv=self.pyramid.parameters["centerv"][index]  # 0-1
        centerh = self.pyramid.parameters["centerh"][index] # 0-1.77777778

        #get polar angle
        polar_angle = np.arctan2(centerh - centerpos[1], centerv - centerpos[0])
        #convert to degrees
        polar_angle = np.degrees(polar_angle)
        #convert to 0-360 degrees
        if polar_angle < 0:
            polar_angle += 360
        return polar_angle



    def index_to_eccentricity(self, index):
        """
        Convert a filter index to eccentricity.
        """
        aspect_ratio = self.pyramid.parameters["aspect_ratio"][0]
        centerpos = (0.5, aspect_ratio/2) # (v,h)
        centerv=self.pyramid.parameters["centerv"][index]  # 0-1
        centerh = self.pyramid.parameters["centerh"][index] # 0-1.77777778
        #get eccentricity
        eccentricity = np.sqrt((centerv - centerpos[0])**2 + (centerh - centerpos[1])**2)
        return eccentricity



    def index_to_spatial_freq(self, index):
        """
        Convert a filter index to spatial frequency.
        """
        return self.pyramid.parameters["spatial_freq"][index]
    
    def index_to_temporal_freq(self, index):
        """
        Convert a filter index to temporal frequency.
        """
        return self.pyramid.parameters["temporal_freq"][index]
    
    def index_to_direction(self, index):
        """
        Convert a filter index to direction.
        """
        return self.pyramid.parameters["direction"][index]

        
    def velocity_from_index(self,index):
        spatial_freq = self.pyramid.parameters["spatial_freq"][index]
        temporal_freq = self.pyramid.parameters["temporal_freq"][index]
        return temporal_freq / spatial_freq   


    def test(self):
        """
        Test the filter info class.
        """
        # Test the filter info class
        pprint(self.pyramid)
        pprint(self.pyramid.parameters)



if __name__ == "__main__":
    # Create a filter info object
    filter_info = FilterInfo()
    # Test the filter info class
    filter_info.test()
    # Print the filter info class
    pprint(filter_info.__dict__)