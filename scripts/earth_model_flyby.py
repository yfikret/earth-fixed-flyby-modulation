"""
Tested with Python 3.12.2

This script computes Earth-fixed land-ocean coupling metrics
from precomputed spacecraft trajectories. Results are deterministic
given identical inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
import csv
from get_land_fraction import land_fraction_fibonacci
from pathlib import Path

CONFIG = {
    "mode": "show_maps",
    # "visualize_trajectory", "record_coupling_constants",
    # "run_plateau_test", "run_drift_test", "run_window_dependence_test"
    # "construct_land_fraction_map", "show_maps"

    "weighting_function": "inverse_r_squared", 
    #"none", "inverse_r", "inverse_r_squared", "inverse_r_cubed", "inverse_r_surface", "inverse_r_surface_squared", 
    
    "flybys": ("GALILEO", "JUNO"),
    #("GALILEO","ROSETTA_2005", "NEAR", "MESSENGER", "CASSINI", "ROSETTA_2009", "JUNO", 
    # "GALILEO_2", "ROSETTA_2007"),

    "max_distance_for_visualizing_and_recording_coupling_constants": 2e4,
    
    "plateau_test_max_distances": (1.0e4, 2.0e4, 5.0e4),
    # (1.0e4, 2.0e4, 5.0e4, 1.0e5, 2.0e5, 5.0e5),

    "window_test_phi_offsets": 36,
    "window_test_max_distance": 2e5, # km

    "fraction_map_degree_resolution": 1,
    "fraction_map_distance": 2e4, # km
    "fraction_map_path": "data_surface/cache/projected_land_fraction.npz",

    "normalization": True,
    "ignore_r_for_land_fraction": False,

    "output_dir" : "results",

    "delta_t": 10,
    "landmask_path": "data_surface/cache/landmask_cache_1deg.npz",

    "all_flybys": ("GALILEO","ROSETTA_2005", "NEAR", "MESSENGER", "CASSINI", "ROSETTA_2009", "JUNO", 
                  "GALILEO_2", "ROSETTA_2007"),

    "plateau_r_km": 2e5,
    "maximum_r_km": 5e5
}

def construct_map():
    Path(CONFIG["fraction_map_path"]).parent.mkdir(parents=True, exist_ok=True)

    print(f"Building projected land fraction map")

    r_km = CONFIG["fraction_map_distance"]
    delta_d = CONFIG["fraction_map_degree_resolution"]

    phi_values = np.deg2rad(np.arange(-180, 181, delta_d))  # 0–360 degrees
    theta_values = np.deg2rad(np.arange(0, 181, delta_d))  # 0–180 degrees
    frac_table = np.zeros((np.size(theta_values), np.size(phi_values)))

    for i, theta_view in enumerate(theta_values):
        #print(f"Current theta view (radians): {theta_view:.3}")
        for j, phi_view in enumerate(phi_values):

            #print("Theta: ", theta_view, " ", "Phi:", phi_view)
            
            frac = land_fraction_fibonacci(theta_view, phi_view, r_km)
            frac_table[i][j] = frac
    
    np.savez(CONFIG["fraction_map_path"],
        frac_table=frac_table)

    print("Saved projected land fraction map")

def show_maps():
     
    # --- Data Loading ---
    landmask_data = np.load(CONFIG["landmask_path"])
    landmask = landmask_data["landmask"]
    lat_vals = landmask_data["lat_vals"]
    lon_vals = landmask_data["lon_vals"]

    projected_land_fraction_data = np.load("data_surface/cache/projected_land_fraction.npz")
    projected_land_fraction = projected_land_fraction_data["frac_table"]

    # --- Global Styling ---
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]

    # --- Subplot 1: Continuous Data (Land Fraction) ---
    im1 = ax.imshow(projected_land_fraction, extent=extent, origin='upper')
    #ax.set_title("Projected Land Fraction", fontsize=20, fontweight='bold')
    ax.set_ylabel("Latitude (degrees)", fontsize=20)
    ax.set_xlabel("Longitude (degrees)", fontsize=20)
    ax.tick_params(labelsize=15)

    # Colorbar for the continuous data
    cbar = fig.colorbar(im1, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Land Fraction', fontsize=20)
    
    plt.show()

    # --- Subplot 2: Binary Data (Landmask) ---
    # We define a discrete colormap: 0=Sea (Blue), 1=Land (Green)
    
    binary_cmap = ListedColormap(["#211347", "#c2df20"]) 
    
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    im2 = ax.imshow(landmask, extent=extent, cmap=binary_cmap, origin='upper')
    #ax.set_title("Binary Landmask", fontsize=20, fontweight='bold')
    ax.set_ylabel("Latitude (degrees)", fontsize=20)
    ax.set_xlabel("Longitude (degrees)", fontsize=20)
    ax.tick_params(labelsize=15)

    # Create a Custom Legend instead of a colorbar
    legend_elements = [
        Patch(facecolor='#211347', label='Sea (0)'),
        Patch(facecolor='#c2df20', label='Land (1)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=15)

    # Final formatting
    plt.tight_layout()
    plt.show()

def visualize_trajectory(phi_array, theta_array, trajectory_values):
    land_fraction_map = np.load(CONFIG["fraction_map_path"])
    projected_land_fraction = land_fraction_map["frac_table"]
    
    H, W = projected_land_fraction.shape
    
    phi_idx = (phi_array + np.pi) * (W / (2 * np.pi))
    phi_i   = np.rint(phi_idx).astype(int) % W

    theta_idx = theta_array * (H / np.pi)
    theta_i = np.clip(np.rint(theta_idx).astype(int), 0, H - 1)
    
    trajectory_values = projected_land_fraction[theta_i, phi_i]

    fig, ax = plt.subplots()
    im1 = ax.imshow(projected_land_fraction)
    cbar = fig.colorbar(im1)
    cbar.ax.tick_params(labelsize=15)
 
    plt.ylabel("Latitude (degrees)", fontsize=20)
    plt.xlabel("Longitude (degrees)", fontsize=20)
    
    plt.scatter(phi_i[0], theta_i[0], c='black', s=150, label="Position at segment start", zorder=2)
    plt.scatter(phi_i[len(phi_i)//2], theta_i[len(theta_i)//2], c='red', s=150, label="Position at periapsis", zorder=2)
    plt.plot(phi_idx, theta_idx, c='white', linewidth=4.0, label = "_nolegend",zorder=1)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=15)
    
   
    #plt.subplot(2, 2, 2)  
    #plt.plot(trajectory_values)
    
    y_offset = 90
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'{-y + y_offset:g}'))
    x_offset = -180
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x + x_offset:g}'))
    plt.show()
    
def get_weighted_land_fractions(path_data):
    radiusEarth = 6.378e3

    position_r, position_phi, position_theta = path_data
    
    trajectory_land_weights = []

    for i in range(position_r.size):
        theta = position_theta[i]
        phi = position_phi[i]
        r = position_r[i]
        trajectory_land_weights.append(land_fraction_fibonacci(theta, phi, r, CONFIG["ignore_r_for_land_fraction"]))

    trajectory_land_weights = np.asarray(trajectory_land_weights)

    if CONFIG["mode"] == "visualize_trajectory":
        visualize_trajectory(position_phi, position_theta, trajectory_land_weights)

    weighting_function = CONFIG["weighting_function"]
    if  weighting_function == "none":
        distance_weighting = 1
    elif weighting_function == "inverse_r":
        distance_weighting = 1 / position_r
    elif weighting_function == "inverse_r_surface":
        distance_weighting = 1 / (position_r - radiusEarth)
    elif weighting_function == "inverse_r_squared":
        distance_weighting = 1 / position_r**2
    elif weighting_function == "inverse_r_surface_squared":
        distance_weighting = 1 / (position_r - radiusEarth)**2
    elif weighting_function == "inverse_r_cubed":
        distance_weighting = 1 / position_r**3
    
    weighted_land_fractions = distance_weighting * trajectory_land_weights 
    return weighted_land_fractions

def get_path_based_asymmetry(weighted_land_fractions, time_array):

    trajectory_size = weighted_land_fractions.size
    
    middle_index = trajectory_size//2
    left_cont = weighted_land_fractions[:middle_index]
    right_cont = weighted_land_fractions[middle_index:]

    time_array_left = time_array[:middle_index]
    time_array_right = time_array[middle_index:]

    left_sum = np.trapezoid(left_cont, time_array_left) 
    right_sum = np.trapezoid(right_cont, time_array_right)
    side_difference = left_sum - right_sum
    side_sum = left_sum + right_sum

    normalized_difference = side_difference / side_sum

    if CONFIG["normalization"]:
        return normalized_difference
    else:
        return side_difference

def get_coupling_value(trajectory_data, max_distance):
    
    position_r_full = trajectory_data["position_mag"]
    position_phi_full = trajectory_data["position_phi"]
    position_theta_full = trajectory_data["position_theta"]

    distance_mask = position_r_full <= max_distance
    position_r = position_r_full[distance_mask]
    position_phi = position_phi_full[distance_mask]
    position_theta = position_theta_full[distance_mask]
    
    trajectory_array_size = len(position_r)
    total_time = CONFIG["delta_t"]*trajectory_array_size
    time_array = np.arange(0, total_time, CONFIG["delta_t"])
    
    path_data = (position_r, position_phi, position_theta)

    weighted_land_fractions = get_weighted_land_fractions(path_data)
    coupling_value = get_path_based_asymmetry(weighted_land_fractions, time_array)
    return coupling_value

def get_coupling_values_for_distance_array(trajectory_data, max_distance_array):
    
    position_r_full = trajectory_data["position_mag"]
    position_phi_full = trajectory_data["position_phi"]
    position_theta_full = trajectory_data["position_theta"]
    
    trajectory_array_size = len(position_r_full)
    total_time = CONFIG["delta_t"]*trajectory_array_size
    time_array = np.arange(0, total_time, CONFIG["delta_t"])
    
    path_data = (position_r_full, position_phi_full, position_theta_full)

    weighted_land_fractions = get_weighted_land_fractions(path_data)
    
    coupling_value_array = []
    for max_distance in max_distance_array:
        distance_mask = position_r_full <= max_distance

        filtered_weighted_land_fractions = weighted_land_fractions[distance_mask]
        filtered_time_array = time_array[distance_mask]
        coupling_value = get_path_based_asymmetry(filtered_weighted_land_fractions, filtered_time_array)
        coupling_value_array.append(coupling_value)

    coupling_value_array = np.asarray(coupling_value_array)
    return coupling_value_array

def show_coupling_value():
    flyby_crafts = CONFIG["flybys"]
    max_distance = CONFIG["max_distance_for_visualizing_and_recording_coupling_constants"]
    
    outdir = Path(CONFIG["output_dir"])
    outdir.mkdir(exist_ok=True)
    weighting_function = CONFIG["weighting_function"]

    results = []
    csv_output_filepath = Path(outdir, f"coupling_values_weighting_{weighting_function}.csv")
    
    for craft in flyby_crafts:
        print(f"Analyzing {craft}")
        craft_data = np.load(f"data/parsed/{craft}.npz", allow_pickle=True)
        trajectory_data = craft_data["trajectory"].item()
        
        coupling_value = get_coupling_value(trajectory_data, max_distance)

        results.append({
            "craft": craft,
            "max_distance_km": max_distance,
            "coupling_value": coupling_value,
            })

        print(craft, coupling_value)
    
    if CONFIG["mode"] == "record_coupling_constants":
        with open(csv_output_filepath, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "craft",
                    "max_distance_km",
                    "coupling_value"
                ],
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved results to {csv_output_filepath}")

def run_drift_test():
    
    all_flybys = CONFIG["all_flybys"]

    weighting_function = CONFIG["weighting_function"]

    drift_left_endpoint = CONFIG["plateau_r_km"]
    drift_right_endpoint = CONFIG["maximum_r_km"]

    max_distance_array = (drift_left_endpoint, drift_right_endpoint)
    
    outdir = Path(CONFIG["output_dir"])
    outdir.mkdir(exist_ok=True)

    csv_output_filepath_drift = Path(outdir, f"distance_graph_drift_weighting_{weighting_function}.csv")

    drift_results = []
    percent_change_array = []

    for craft in all_flybys:
        
        craft_data = np.load(f"data/parsed/{craft}.npz", allow_pickle=True)
        trajectory_data = craft_data["trajectory"].item()
        
        coupling_value_array = get_coupling_values_for_distance_array(trajectory_data, max_distance_array)
        
        delta_coupling = coupling_value_array[1] - coupling_value_array[0]
        percent_change = 100*(delta_coupling/coupling_value_array[0])

        percent_change_array.append(percent_change)

        drift_results.append({
            "craft": craft,
            "plateau_cutoff": drift_left_endpoint,
            "maximum_r": drift_right_endpoint,
            "coupling_value_at_plateau_cutoff": coupling_value_array[0],
            "coupling_value_at_r_max": coupling_value_array[1],
            "percent_change_across_range": percent_change
            }) 
        
        absolute_average_percent_change = np.mean(np.abs(np.asarray(percent_change_array)))
        drift_results.append({
            "craft": "Absolute average",
            "plateau_cutoff": drift_left_endpoint,
            "maximum_r": drift_right_endpoint,
            "coupling_value_at_plateau_cutoff": coupling_value_array[0],
            "coupling_value_at_r_max": coupling_value_array[1],
            "percent_change_across_range": absolute_average_percent_change
            }) 
        
        print(f"Acquired drift for {craft}") 

    with open(csv_output_filepath_drift, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "craft",
                "plateau_cutoff",
                "maximum_r",
                "coupling_value_at_plateau_cutoff",
                "coupling_value_at_r_max",
                "percent_change_across_range"
            ],
        )
        writer.writeheader()
        writer.writerows(drift_results)

    print(f"Saved results to {csv_output_filepath_drift}")

def run_plateau_test():

    all_flybys = CONFIG["all_flybys"]

    positive_main = ("GALILEO", "NEAR", "ROSETTA_2005")          # 3 main positive
    positive_uncertain = ("ROSETTA_2007", "GALILEO_2")            # 2 uncertain positive
    negative_flybys = ("JUNO", "ROSETTA_2009", "MESSENGER", "CASSINI")  # 4 negative

    weighting_function = CONFIG["weighting_function"]

    #max_distance_array = CONFIG["plateau_test_max_distances"]

    max_distance_array = np.unique(np.concatenate([
        np.logspace(np.log10(1e4), np.log10(1.5e5), 20),
        np.linspace(1.5e5, 5e5, 12)])).astype(float)
    
    outdir = Path(CONFIG["output_dir"])
    outdir.mkdir(exist_ok=True)

    csv_output_filepath = Path(outdir, f"distance_graph_weighting_{weighting_function}.csv")
    image_output_filepath = Path(outdir, f"distance_graph_weighting_{weighting_function}.png")

    results = []
    
    # --- Plot setup ---
    fig, axes = plt.subplots(
        nrows=2, ncols=1, figsize=(7.5, 8.5), sharex=True, constrained_layout=True
    )

    # Common axis styling
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", linewidth=0.6, alpha=0.35)
        ax.axvline(CONFIG["plateau_r_km"], linestyle="--", linewidth=1.2)
        ax.set_ylabel("Coupling proxy $C(r_{\\max})$")
    axes[-1].set_xlabel("Integration boundary $r_{\\max}$ (km)")

    axes[0].set_title("Positive-$C$ flybys (including uncertain positives)")
    axes[1].set_title("Negative-$C$ flybys")

    colors = plt.get_cmap("tab10").colors
    
    max_distance_array_x = np.asarray(max_distance_array)
    for craft_index, craft in enumerate(all_flybys):

        craft_data = np.load(f"data/parsed/{craft}.npz", allow_pickle=True)
        trajectory_data = craft_data["trajectory"].item()
        #trajectory_data = sparsify_indices(trajectory_data)

        coupling_value_array = get_coupling_values_for_distance_array(trajectory_data, max_distance_array)
        
        for distance_index, max_distance in enumerate(max_distance_array):
            
            coupling_value = coupling_value_array[distance_index]
            results.append({
            "craft": craft,
            "max_distance_km": max_distance,
            "coupling_value": coupling_value,
            })

        color = colors[craft_index]

        if craft in positive_main:
            axes[0].plot(max_distance_array_x, coupling_value_array, color=color, label=craft, linestyle="-", linewidth=2.0, alpha=0.95)
        elif craft in positive_uncertain:
            axes[0].plot(max_distance_array_x, coupling_value_array, color=color, label=craft, linestyle="--", linewidth=1.8, alpha=0.85)
        elif craft in negative_flybys:
            axes[1].plot(max_distance_array_x, coupling_value_array, color=color, label=craft, linestyle="-", linewidth=2.0, alpha=0.95)
        
        print(f"Processed {craft}")
        #plt.plot(max_distance_array, coupling_value_array, color=colors[i % 10])   

    for ax in axes:
        ax.legend(loc="best", fontsize="medium", frameon=True, ncol=2)

    # Optional: annotate the cutoff line
    axes[0].annotate(
        r"chosen cutoff $2\times10^5$ km",
        xy=(CONFIG["plateau_r_km"], 0.98),
        xycoords=("data", "axes fraction"),
        xytext=(6, -8),
        textcoords="offset points",
        rotation=90,
        va="top",
        fontsize="medium",
    )
    plt.savefig(image_output_filepath)
    plt.show()
    plt.clf()
   
    with open(csv_output_filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "craft",
                "max_distance_km",
                "coupling_value"
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {csv_output_filepath}")

def run_window_dependence_test():

    results = []
    outdir = Path(CONFIG["output_dir"])
    outdir.mkdir(exist_ok=True)
    
    weighting_function = CONFIG["weighting_function"]

    csv_output_filepath = Path(outdir, f"window_dependence_weighting_{weighting_function}.csv")

    flyby_crafts = CONFIG["flybys"]

    max_distance_selection = CONFIG["window_test_max_distance"]

    offset_total = CONFIG["window_test_phi_offsets"]
    offset_unit = 2*np.pi/offset_total

    all_coupling_signs = []
    for phi_offset_index in range(offset_total):
        print(f"Phi offset: {phi_offset_index + 1} of {offset_total}")
        
        phi_offset = offset_unit*phi_offset_index

        coupling_values_for_offset = []

        for craft in flyby_crafts:
            
            craft_data = np.load(f"data/parsed/{craft}.npz", allow_pickle=True)
            trajectory_data = craft_data["trajectory"].item()

            trajectory_data = trajectory_data.copy()
            trajectory_data["position_phi"] = trajectory_data["position_phi"] + phi_offset

            coupling_value = get_coupling_value(trajectory_data, max_distance_selection)
            coupling_values_for_offset.append(coupling_value)

            results.append({
                "phi_offset_index": phi_offset_index,
                "phi_offset_rad": phi_offset,
                "craft": craft,
                "coupling_value": coupling_value,
                "coupling_sign": int(np.sign(coupling_value)),
            })

        all_coupling_signs.append(np.sign(coupling_values_for_offset))

    true_signs = all_coupling_signs[0]

    indices_of_match = []
    for phi_offset_index in range(offset_total):
        sign_row = all_coupling_signs[phi_offset_index]
        if (sign_row==true_signs).all():
            indices_of_match.append(phi_offset_index)

    match_rate = len(indices_of_match) / offset_total
    print("True signs: ", true_signs)
    print("Matching indices: ", indices_of_match)
    print("Total matching indices: ", len(indices_of_match))
    print(f"Match rate: {match_rate:.3}")

    with open(csv_output_filepath, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phi_offset_index",
                "phi_offset_rad",
                "craft",
                "coupling_value",
                "coupling_sign",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {csv_output_filepath}")

def main():
    
    if CONFIG["mode"] == "visualize_trajectory":
        print("Running trajectory visualization mode")
        if not Path(CONFIG["fraction_map_path"]).exists():
            raise FileNotFoundError("Projected land fraction map not found. Run construct_map() first.")
        show_coupling_value()
    elif CONFIG["mode"] == "record_coupling_constants":
        show_coupling_value()
    elif CONFIG["mode"] == "show_maps":
        if not Path(CONFIG["fraction_map_path"]).exists():
            raise FileNotFoundError("Projected land fraction map not found. Run construct_map() first.")
        
        show_maps()
    elif CONFIG["mode"] == "construct_land_fraction_map":
        construct_map()
    elif CONFIG["mode"] == "run_plateau_test":  
        print("Running saturation test")
        run_plateau_test()
    elif CONFIG["mode"] == "run_drift_test":  
        print("Running drift test")
        run_drift_test()
    elif CONFIG["mode"] == "run_window_dependence_test":  
        print("Running window test")
        run_window_dependence_test()
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()


