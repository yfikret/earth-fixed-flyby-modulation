import spiceypy as sp
import numpy as np
from pathlib import Path

CONFIG = {
    "output_dir": "data/parsed",
    "kernel_dir": "data/kernels",
    "metakernel": "flyby-main.tm",
    "dt_seconds": 10,
    "distance_limit_km": 6e5, # 0.9e6, # SPICE position/velocity units: km and km/s
}

def get_spherical_position(x_vec):
    x, y, z    = x_vec
    
    rho = np.hypot(x, y)            # = sqrt(x² + y²)
    
    theta_x = np.arctan2(rho, z)      # returns [0, π], no clamp needed\
    phi_x   = np.arctan2(y, x)        # returns (−π, +π]

    return theta_x, phi_x

def search_time(direction_option, sc_id, et_start, body_id, frame, distance_limit, dt):
    """
    Return time samples where the spacecraft stays within ±distance_limit
    (radial displacement) from the position at et_start or et_end
    """

    if direction_option == "forward":
        dt_multiple = 1
    elif direction_option == "reverse":
        dt_multiple = -1

    # Search in time
    et = et_start
    while True:
        et += dt_multiple*dt
        r, _ = sp.spkezr(sc_id, et, frame, 'NONE', body_id)
        r = np.array(r[:3])

        if np.linalg.norm(r) > distance_limit:
            et_found = et
            break
    
    return et_found

def get_trajectory_data(
    sc_id: str,
    times_et: np.ndarray,
    earth_id: str,
    frame: str = "ITRF93"
) -> dict:
    # Generate time samples 
    
    r_km_array = []
    position_phi_array = []
    position_theta_array = []
    #velocity_mag_array = []
    #velocity_theta_array = []

    # Parse ITRF93 coordinates

    for et in times_et:

        # Spacecraft state relative to Earth
        state_sc_earth, _ = sp.spkezr(sc_id, et, frame, 'NONE', earth_id)
        
        position = np.array(state_sc_earth[:3])
        position_mag = np.linalg.norm(position)

        #velocity = np.array(state_sc_earth[3:6])
        #velocity_mag = np.linalg.norm(velocity)

        position_theta, position_phi = get_spherical_position(position)

        r_km_array.append(position_mag)
        position_phi_array.append(position_phi)
        position_theta_array.append(position_theta)
        #velocity_mag_array.append(velocity_mag)
        #velocity_theta_array.append(velocity_theta)
        
    r_km_array = np.asarray(r_km_array, dtype=float)
    position_phi_array = np.asarray(position_phi_array, dtype=float)
    position_theta_array = np.asarray(position_theta_array, dtype=float)
    #velocity_mag_array = np.asarray(velocity_mag_array, dtype=float)
    #velocity_theta_array = np.asarray(velocity_theta_array, dtype=float)
    
    return {
            "position_mag":     r_km_array,
            "position_phi":     position_phi_array,
            "position_theta":   position_theta_array
            }

def write_trajectory_data():

    # --- Load SPICE Meta-Kernel ---

    kernel_dir = Path(CONFIG["kernel_dir"])
    metakernel_dir = Path(kernel_dir, CONFIG["metakernel"])
    sp.furnsh(str(metakernel_dir))
    
    # --- Constants ---

    DT = CONFIG["dt_seconds"]   # Sampling interval [s]

    # SPICE ID codes 
    EARTH_ID = "399"
    GALILEO_ID = "-77"
    CASSINI_ID = "-82"
    ROSETTA_ID = "-226"
    MESSENGER_ID = "-236"
    NEAR_ID = "-93"
    JUNO_ID = "-61"

    # Flyby definitions: spacecraft IDs and approximate periapsis times (UTC)

    flyby_parameters = [
        {'name' : "GALILEO", 'sc_id': GALILEO_ID, 'peri_utc': '1990-12-08T20:34:30'},
        {'name' : "GALILEO_2", 'sc_id': GALILEO_ID, 'peri_utc': '1992-12-08T15:09:20'},
        {'name' : "CASSINI", 'sc_id': CASSINI_ID, 'peri_utc': '1999-08-18T03:28:30'},
        {'name' : "ROSETTA_2005", 'sc_id': ROSETTA_ID, 'peri_utc': '2005-03-04T22:09:10'},
        {'name' : "ROSETTA_2007", 'sc_id': ROSETTA_ID, 'peri_utc': '2007-11-13T20:57:20'},
        {'name' : "ROSETTA_2009", 'sc_id': ROSETTA_ID, 'peri_utc': '2009-11-13T07:45:40'},
        {'name' : "NEAR", 'sc_id': NEAR_ID, 'peri_utc': '1998-01-23T07:23:00'},
        {'name' : "MESSENGER", 'sc_id': MESSENGER_ID, 'peri_utc': '2005-08-02T19:13:10'},
        {'name' : "JUNO", 'sc_id': JUNO_ID, 'peri_utc': '2013-10-09T19:21:20'},
    ]

    distance_limit = CONFIG["distance_limit_km"] 

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    
    for parameter_set in flyby_parameters:
        
        SC_NAME = parameter_set['name']
        SC_ID    = parameter_set['sc_id']
        FLYBY_ET = sp.str2et(parameter_set['peri_utc'])
        
        out_npz = Path(CONFIG["output_dir"], f"{SC_NAME}.npz")
        print(f"Writing {out_npz}")

        et_min = search_time("reverse", SC_ID, FLYBY_ET, EARTH_ID, "J2000", distance_limit, DT)
        et_max = search_time("forward", SC_ID, FLYBY_ET, EARTH_ID, "J2000", distance_limit, DT)
                
        start_utc = sp.et2utc(et_min, "C", 3)
        end_utc = sp.et2utc(et_max, "C", 3)

        times = np.arange(et_min, et_max, DT)
        
        # Parse J2000 coordinates

        trajectory = get_trajectory_data(SC_ID, times, EARTH_ID)

        metadata = {
                        "spacecraft":       SC_NAME,
                        "dt_seconds":       DT,
                        "distance_limit_km":distance_limit,
                        "peri_utc":         parameter_set["peri_utc"],
                        "start_utc":        start_utc,
                        "end_utc":          end_utc,          
                        "kernel_metafile":  CONFIG["metakernel"]
                    }
 
        np.savez(out_npz, times=times, trajectory=trajectory, metadata=metadata)
    
    print("Sample extraction complete. Records saved for each flyby.")

if __name__ == "__main__":
    write_trajectory_data()