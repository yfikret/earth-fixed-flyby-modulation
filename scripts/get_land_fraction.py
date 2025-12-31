import numpy as np
import matplotlib.pyplot as plt

CONFIG = {
    "radius_proportion": 1.00,
    "projection_resolution": 100,
    "input_path": "data_surface/cache/landmask_cache_1deg.npz",
}

R_EARTH = 6371.0

landmask_data = np.load(CONFIG["input_path"])
landmask = landmask_data["landmask"]
lat_vals = landmask_data["lat_vals"]
lon_vals = landmask_data["lon_vals"]

def sph_to_cart(theta, phi):
    """
    converts angles into a 3D unit vector v, which represents the "center" of the camera's view
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

def build_projection_frame(v):
    """
    generates two vectors, e1 and e2, that are perpendicular to the view vector v
    """

    if abs(v[2]) < 0.9:
        tmp = np.array([0, 0, 1])
    else:
        tmp = np.array([1, 0, 0])

    e1 = np.cross(v, tmp)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(v, e1)
    e2 /= np.linalg.norm(e2)

    return e1, e2

import numpy as np
def land_fraction_fibonacci(theta_view, phi_view, r_km, ignore_r = False, samples=1000):
    """
    Calculates land fraction using a deterministic Fibonacci Lattice (Golden Spiral).
    """

    indices = np.arange(samples) + 0.5 # Offset to avoid poles

    if not ignore_r:
        # 1. Calculate the Horizon Limit
        # If altitude is 0, we see nothing (limit -> 1). 
        # If altitude is infinite, we see hemisphere (limit -> 0).
        cos_alpha = R_EARTH / r_km
        
        # 2. Generate Fibonacci Points directly on the Visible Cap
        # We want 'samples' points distributed uniformly in Area.
        # On a sphere, Area is proportional to Height (z).
        # So we linearly space z from cos_alpha (horizon) to 1.0 (nadir).
        
        # z_local goes linearly from Horizon to Nadir
        z_local = cos_alpha + (1 - cos_alpha) * indices / samples
    else:
        z_local = indices / samples
    
    # The Golden Angle distributes points along the spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta_local = golden_angle * indices  # This acts as our 'phi' around the axis

    # Convert to Cartesian in Local Frame
    r_local = np.sqrt(1 - z_local**2)
    x_local = r_local * np.cos(theta_local)
    y_local = r_local * np.sin(theta_local)

    # 3. Rotate to World Frame
    v = sph_to_cart(theta_view, phi_view)
    e1, e2 = build_projection_frame(v)

    # Broadcasting rotation
    p = (x_local[:, np.newaxis] * e1 + 
         y_local[:, np.newaxis] * e2 + 
         z_local[:, np.newaxis] * v)
    
    p /= np.linalg.norm(p, axis=1)[:, np.newaxis]

    # 4. Map to Lat/Lon
    world_theta = np.arccos(np.clip(p[:, 2], -1, 1))
    world_phi   = np.arctan2(p[:, 1], p[:, 0])

    colat_deg = np.rad2deg(world_theta)
    lon_deg = np.rad2deg(world_phi)

    ii = np.round(colat_deg - 0.5).astype(int)
    jj = (np.round(lon_deg + 179.5).astype(int)) % 360
    
    # Clip to be safe within array bounds (assuming 180x360 grid)
    ii = np.clip(ii, 0, 179)
    jj = np.clip(jj, 0, 359)
    
    # Extract data
    is_land = landmask[ii, jj]
    
    return np.mean(is_land)

def test_speed():
     trajectory_fractions = []

     windows = 1
     craft_count = 8
     
     for window in range(windows):
          for craft in range(craft_count):
               time_count = 20000
               traj_lats = np.ones(time_count)*0.3
               traj_lons = np.ones(time_count)*0.2
               traj_alts = np.ones(time_count)*0.1

               # Example loop
               for i in range(len(traj_lats)):
                    # Convert lat/lon to radians for the view direction
                    theta_view = np.deg2rad(90 - traj_lats[i])
                    phi_view   = np.deg2rad(traj_lons[i])
               
               frac = land_fraction_fibonacci(theta_view, phi_view, traj_alts[i])
               trajectory_fractions.append(frac)





#show_maps()



