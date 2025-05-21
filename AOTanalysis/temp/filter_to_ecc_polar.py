import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# from robert's code useful for converting filter coordinates to polar coordinates and ecc

# https://github.com/robsatz/aot-analysis/blob/b91650fe98099cf3e9ece5894b3e696476bf70d1/aot_analysis/motion_energy/visualize.py


# def load_filters(screen_size_cm, screen_distance_cm):
#     with open(DIR_MOTION_ENERGY / "pyramid.pkl", "rb") as f:
#         pyramid = pickle.load(f)

#     filters = pd.DataFrame(pyramid.filters)

#     # take screen_size_cm as width, calculate pixel size in cm
#     vdim_px, hdim_px, _ = pyramid.definition.stimulus_vht_fov
#     px_size_cm = screen_size_cm / hdim_px

#     # pymoten encodes coordinates as distances from top left corner relative to pixel **height**
#     # recode to absolute distances from center in cm
#     centerv_px = vdim_px * filters.loc[:, "centerv"] - vdim_px / 2
#     centerh_px = vdim_px * filters.loc[:, "centerh"] - hdim_px / 2

#     centerv_cm = centerv_px * px_size_cm
#     centerh_cm = centerh_px * px_size_cm

#     # convert dims to degrees of visual angle
#     filters["centerv"] = 2.0 * np.degrees(
#         np.arctan(centerv_cm / (2.0 * screen_distance_cm))
#     )
#     filters["centerh"] = 2.0 * np.degrees(
#         np.arctan(centerh_cm / (2.0 * screen_distance_cm))
#     )

#     # descriptive stats: nanmin and nanmax
#     print(f'x_deg: {np.nanmin(filters["centerh"])} - {np.nanmax(filters["centerh"])}')
#     print(f'y_deg: {np.nanmin(filters["centerv"])} - {np.nanmax(filters["centerv"])}')

#     return filters


# def cart2polar(x_deg, y_deg):
#     # descriptive stats: nanmin and nanmax
#     print(f"x_deg: {np.nanmin(x_deg)} - {np.nanmax(x_deg)}")
#     print(f"y_deg: {np.nanmin(y_deg)} - {np.nanmax(y_deg)}")

#     # assumes coordinates in degrees of visual angle
#     ecc = np.sqrt(x_deg**2 + y_deg**2)
#     polar = np.angle(x_deg + y_deg * 1j)
#     # descriptive stats: nanmin and nanmax
#     print(f"ecc: {np.nanmin(ecc)} - {np.nanmax(ecc)}")
#     print(f"polar: {np.nanmin(polar)}" + f"- {np.nanmax(polar)}")
#     return pd.DataFrame({"ecc": ecc, "polar": polar})
