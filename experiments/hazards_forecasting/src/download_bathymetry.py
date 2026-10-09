"""ETOPO 2022 (NOAA NCEI, 60 arc-second bedrock elevation) subsampled to 5 arc-minutes through OPeNDAP, no login.
Saves data/bathy/etopo_5min.npz (z in metres, lat, lon). The parameters of the physical tsunami expert come from this grid."""
import re
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "bathy" / "etopo_5min.npz"
URL = ("https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc.ascii"
       "?z%5B{a}:5:{b}%5D%5B0:5:21599%5D")


def band(a, b):
    for k in range(4):
        try:
            txt = urllib.request.urlopen(URL.format(a=a, b=b), timeout=300).read().decode()
            break
        except Exception:  # noqa: BLE001
            if k == 3:
                raise
    sect = txt.split("z.z[")[1].split("z.lat[")[0]
    rows = [r for r in sect.split("\n")[1:] if r.startswith("[")]
    z = np.array([[float(v) for v in r.split(",")[1:]] for r in rows], dtype=np.float32)
    lat = np.array([float(v) for v in txt.split("z.lat[")[1].split("\n")[1].split(",")])
    return z, lat


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    zs, lats = [], []
    for a in range(0, 10800, 900):
        z, lat = band(a, min(a + 899, 10799))
        zs.append(z); lats.append(lat)
        print("band", a, z.shape, flush=True)
    z, lat = np.vstack(zs), np.concatenate(lats)
    lon = np.linspace(-180 + 1 / 120, 180 - 1 / 120, 21600)[::5]
    np.savez_compressed(OUT, z=z, lat=lat, lon=lon[: z.shape[1]])
    print(z.shape, float(z.min()), float(z.max()))
