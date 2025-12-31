import numpy as np
import fiona
from shapely.geometry import shape, Point
from shapely.prepared import prep
from shapely.strtree import STRtree
from pathlib import Path

CONFIG = {
    "output_dir": "data_surface/cache",
    "output_path": "data_surface/cache/landmask_cache_1deg.npz",
    "input_path": "data_surface/ne_110m_land/ne_110m_land.shp",
}

def build_landmask(shp_path=CONFIG["input_path"],
                   dlat=1.0, dlon=1.0,
                   out_npz=CONFIG["output_path"]):
    # Grid centers
    lat_vals = np.arange(90 + dlat/2,  -90, -dlat)   # e.g., -89.5..89.5
    lon_vals = np.arange(-180 + dlon/2, 180, dlon)  # e.g., -179.5..179.5

    geoms = []
    with fiona.open(shp_path) as src:
        for feat in src:
            geoms.append(shape(feat["geometry"]))

    prepared = [prep(g) for g in geoms]
    tree = STRtree(geoms)

    landmask = np.zeros((len(lat_vals), len(lon_vals)), dtype=np.uint8)

    for i, lat in enumerate(lat_vals):
        for j, lon in enumerate(lon_vals):
            p = Point(lon, lat)
            for idx in tree.query(p):
                if prepared[idx].contains(p):
                    landmask[i, j] = 1
                    break

    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)

    np.savez(out_npz, landmask=landmask, lat_vals=lat_vals, lon_vals=lon_vals)
    print(f"Saved {out_npz}")
    return out_npz

if __name__ == "__main__":
    build_landmask()