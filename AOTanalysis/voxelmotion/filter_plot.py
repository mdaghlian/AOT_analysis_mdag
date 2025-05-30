import os
import numpy as np
import matplotlib.pyplot as plt


# ************************************
# GENERAL PURPOSE
def plot_centered_imshow(ax, mat, x, y, size, **kwargs):
    """
    Plot an image (matrix) centered at (x, y) with the given size (width, height).

    Parameters:
    ax   -- matplotlib axis to plot on
    mat  -- 2D numpy array (n x m) representing the image
    x, y -- coordinates for the center of the image
    size -- tuple (width, height) specifying the size of the image
    """
    width, height = size
    extent = [x - width / 2, x + width / 2, y - height / 2, y + height / 2]
    ax.imshow(mat, extent=extent, origin='lower', aspect='auto', **kwargs)

# ************************************
# BASED NISHIMOTO SELECTIVITY



# ************************************
# BASED ON PURE WEIGHTS 
def plot_tf_sf_maps(filt_info, weights, vmin=None, vmax=None, cmap='viridis', pc=99,):
    """
    Plot TF x SF maps as small images located at each (x, y) filter position.

    Parameters:
        filt_info: an instance of FilterObj
        weights: (n_filters,) array of weights
        vmin, vmax: color scale limits
        cmap: colormap for the images
        show_axes: whether to show axis labels and ticks
    """
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights[:, None]

    df = filt_info.filter_df
    x_vals = df['x'].to_numpy()
    y_vals = df['y'].to_numpy()
    sf_vals = df['SF'].to_numpy()
    tf_vals = df['TF'].to_numpy()

    unique_sf = np.sort(df['SF'].unique())
    unique_tf = np.sort(df['TF'].unique())
    n_sf = len(unique_sf)
    n_tf = len(unique_tf)

    xy = np.stack([x_vals, y_vals], axis=1)
    xy_unique, xy_idx = np.unique(xy, axis=0, return_inverse=True)
    n_locs = len(xy_unique)

    all_mats = []
    for loc in range(n_locs):
        loc_mask = (xy_idx == loc)
        sf_sub = sf_vals[loc_mask]
        tf_sub = tf_vals[loc_mask]

        mat = np.zeros((n_tf, n_sf))
        overall_weights = np.zeros(n_locs)
        for i, sf in enumerate(unique_sf):
            for j, tf in enumerate(unique_tf):
                match = (sf_sub == sf) & (tf_sub == tf)
                if np.any(match):
                    mat[j, i] = np.mean(weights[loc_mask][match])
                    overall_weights[loc] += np.sum(weights[loc_mask][match])
        all_mats.append((xy_unique[loc], overall_weights[loc], mat))

    # PLOTTING
    # round x and y for limits
    xmin, xmax = np.floor(xy_unique[:, 0].min())-0.5, np.ceil(xy_unique[:, 0].max())+0.5
    ymin, ymax = np.floor(xy_unique[:, 1].min())-0.5, np.ceil(xy_unique[:, 1].max())+0.5    
    # Fig + ax, with correct aspect ratio
    fig, ax = plt.subplots(figsize=(8*filt_info.aspect_ratio, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # top n percentile of s
    overall_weight_pc = np.percentile(
        [s for _, s, _ in all_mats], pc
    )
    print(f"Overall weight {pc}th percentile:{overall_weight_pc:.3f}")
    for (x, y),s, mat in all_mats:
        if s < overall_weight_pc:
            continue
        plot_centered_imshow(
            ax, mat, x, y,
            size=(0.1*n_sf, 0.1*n_tf),
            cmap=cmap, vmin=vmin, vmax=vmax
        )

    plt.title('TF x SF Weight Maps Plotted at (x, y) Locations')
    # plt.colorbar(sm, ax=ax, label='Collapsed Weights')
    plt.show()




def plot_tf_sf_maps_with_dir_selectivity(
    filt_info,
    weights,
    vmin=None,
    vmax=None,
    cmap='viridis',
    percentile_threshold=99,
    arrow_scale=0.5
):
    """
    Plot TF vs SF weight maps at each spatial filter position,
    with arrows indicating mean motion direction weighted by selectivity.

    Steps:
      1. Extract filter metadata (x, y, spatial freq, temporal freq, direction)
      2. Identify unique positions and create one TF×SF map per position.
      3. Compute per-position:
         - TF×SF weight matrix (average if multiple weights)
         - Direction selectivity: vector sum magnitude normalized by L2 weight norm
         - Mean direction as complex phase
         - Total weight norm (L2 norm)
      4. Determine global color scale (vmin/vmax) if not provided.
      5. Threshold locations by weight percentile to focus on strong filters.
      6. Plot each location:
         - TF×SF map via imshow centered at (x,y)
         - Scatter dot colored by weight norm
         - Arrow for mean direction scaled by selectivity
      7. Add colorbars and title for clarity.

    Parameters:
        filt_info           : FilterInfoV2 instance containing filter_df
        weights             : array of shape (n_filters,) or (n_filters,1)
        vmin, vmax          : global color limits for maps
        cmap                : colormap for TF×SF maps
        percentile_threshold: percentile for total-weight thresholding
        arrow_scale         : scale factor for arrow length
    """
    # Ensure weights is 2D: (n_filters, 1)
    weights = np.asarray(weights).reshape(-1, 1)

    # Extract filter attributes from DataFrame
    df = filt_info.filter_df
    x_coords = df['x'].to_numpy()
    y_coords = df['y'].to_numpy()
    sf = df['SF'].to_numpy()          # spatial frequency
    tf = df['TF'].to_numpy()          # temporal frequency
    direction_deg = df['dir'].to_numpy()  # preferred direction (degrees)

    # Unique SF/TF axes for matrices
    sf_unique = np.sort(df['SF'].unique())
    tf_unique = np.sort(df['TF'].unique())
    n_sf, n_tf = len(sf_unique), len(tf_unique)

    # Unique spatial locations
    xy = np.stack([x_coords, y_coords], axis=1)
    xy_unique, xy_idx = np.unique(xy, axis=0, return_inverse=True)
    n_positions = xy_unique.shape[0]

    # Containers for per-position results
    tf_sf_maps = []         # list of (n_tf x n_sf) arrays
    mean_dirs = []          # list of (dx, dy) unit vectors
    dir_selectivity = []    # magnitude [0..1]
    weight_norms = []       # L2 norm of weights per position

    # Build per-location summaries
    for pos in range(n_positions):
        mask = (xy_idx == pos)
        sf_sub = sf[mask]
        tf_sub = tf[mask]
        dir_sub = direction_deg[mask]
        w_sub = weights[mask].flatten()

        # Initialize TF×SF weight matrix (rows=tf, cols=sf)
        mat = np.zeros((n_tf, n_sf))
        for i_s, s in enumerate(sf_unique):
            for j_t, t in enumerate(tf_unique):
                sel = (sf_sub == s) & (tf_sub == t)
                if np.any(sel):
                    mat[j_t, i_s] = np.nanmean(w_sub[sel])

        # Flatten NaNs to zero for plotting
        mat = np.nan_to_num(mat, nan=0.0)

        # Compute vector sum in complex plane for direction
        radians = np.deg2rad(dir_sub)
        vec = np.sum(w_sub * np.exp(1j * radians))
        l2_norm = np.linalg.norm(w_sub)
        # Direction selectivity: normalized magnitude of vector sum
        sel_strength = np.abs(vec) / l2_norm if l2_norm > 0 else 0.0
        # Unit vector components for mean direction
        angle = np.angle(vec)
        dx, dy = np.cos(angle), np.sin(angle)

        # Save
        tf_sf_maps.append(mat)
        mean_dirs.append((dx, dy))
        dir_selectivity.append(sel_strength)
        weight_norms.append(l2_norm)

    # Determine global vmin/vmax if not user-specified
    all_vals = np.concatenate([m.ravel() for m in tf_sf_maps])
    if vmin is None:
        vmin = np.min(all_vals)
    if vmax is None:
        vmax = np.max(all_vals)

    # Threshold on weight norm percentile
    threshold = np.percentile(weight_norms, percentile_threshold)
    print(f"Using weight norm {percentile_threshold}th percentile = {threshold:.3f}")

    # Set up plot limits
    xmin, xmax = np.floor(xy_unique[:,0].min()) - 0.5, np.ceil(xy_unique[:,0].max()) + 0.5
    ymin, ymax = np.floor(xy_unique[:,1].min()) - 0.5, np.ceil(xy_unique[:,1].max()) + 0.5

    fig, ax = plt.subplots(figsize=(8 * filt_info.aspect_ratio, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', 'box')

    # Plot each position above threshold
    scatter_vals = []
    scatter_pos = []
    for idx, (pos, norm) in enumerate(zip(xy_unique, weight_norms)):
        if norm < threshold:
            continue
        x0, y0 = pos
        # Plot TF×SF map as small image
        plot_centered_imshow(
            ax,
            tf_sf_maps[idx],
            x0,
            y0,
            size=(0.1 * n_sf, 0.1 * n_tf),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        # Collect for scatter
        scatter_pos.append((x0, y0))
        scatter_vals.append(norm)

        # Plot arrow for direction selectivity
        dx, dy = mean_dirs[idx]
        strength = dir_selectivity[idx]
        ax.arrow(
            x0, y0,
            dx * strength * arrow_scale,
            dy * strength * arrow_scale,
            head_width=0.25,
            head_length=0.05,
            fc='red', ec='red',
            length_includes_head=True
        )

    # Scatter all filtered points colored by norm
    scatter_pos = np.array(scatter_pos)
    scatter_vals = np.array(scatter_vals)
    sc = ax.scatter(
        scatter_pos[:,0], scatter_pos[:,1],
        c=scatter_vals, cmap='plasma',
        vmin=min(weight_norms), vmax=max(weight_norms),
        marker='o', edgecolor='k'
    )
    # Colorbar for scatter magnitude
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('Weight L2 Norm')

    # Global colorbar for TF×SF maps
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(all_vals)
    mappable.set_clim(vmin, vmax)
    cbar2 = fig.colorbar(mappable, ax=ax, pad=0.08)
    cbar2.set_label('TF×SF Weight')

    ax.set_title('TF×SF Weight Maps with Direction Selectivity Arrows')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    plt.tight_layout()
    plt.show()
