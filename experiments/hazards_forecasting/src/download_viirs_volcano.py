"""Daily VIIRS (Suomi NPP, 375 m) thermal-anomaly pixel counts around active volcanoes, from NASA GIBS WMS (no login).
One request per volcano-day (per-volcano box, 400x400 px); counts the pixels the layer draws as active-fire/thermal
detections. Cached per volcano in data/volcano/viirs_<name>.csv; safe to re-run (resumes)."""
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "data" / "volcano"
# (lat, lon, half-width of the box in degrees). Narrow boxes keep vegetation fires out of the series; Kilauea needs 0.2
# because Pu'u O'o is 0.18 deg east of the summit.
VOLC = {"kilauea": (19.42, -155.29, 0.20), "etna": (37.75, 14.99, 0.10), "nyiragongo": (-1.52, 29.25, 0.04),
        "erta_ale": (13.60, 40.67, 0.05), "sakurajima": (31.58, 130.66, 0.05)}
URL = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=EPSG:4326"
       "&FORMAT=image/png&TRANSPARENT=TRUE&WIDTH=400&HEIGHT=400&LAYERS=VIIRS_SNPP_Thermal_Anomalies_375m_All"
       "&BBOX={s},{w},{n},{e}&TIME={d}")


def count(args):
    (lat, lon, hw), d = args
    url = URL.format(s=lat - hw, n=lat + hw, w=lon - hw, e=lon + hw, d=d)
    for k in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                a = np.array(Image.open(io.BytesIO(r.read())).convert("RGBA"))
            return d, int((a[..., 3] > 0).sum())
        except Exception:  # noqa: BLE001
            time.sleep(2 * (k + 1))
    return d, -1


if __name__ == "__main__":
    d0, d1 = date(2012, 3, 1), date(2025, 12, 31)
    days = [(d0 + timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    only = sys.argv[1:] or list(VOLC)
    for name, ll in ((k, v) for k, v in VOLC.items() if k in only):
        f = OUT / f"viirs_{name}.csv"
        done = {}
        if f.exists():
            done = {l.split(",")[0]: int(l.split(",")[1]) for l in f.read_text().split("\n")[1:] if l}
        todo = [(ll, d) for d in days if done.get(d, -1) < 0]
        with ThreadPoolExecutor(24) as ex:
            for d, c in ex.map(count, todo):
                done[d] = c
        f.write_text("date,pixels\n" + "\n".join(f"{d},{done[d]}" for d in sorted(done)))
        print(name, len(done), "days, failed", sum(v < 0 for v in done.values()), flush=True)
