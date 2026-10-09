"""NASA GIBS WMS snapshots (public satellite imagery, no login), cached in data/gibs/."""
import hashlib
import io
import urllib.request

import numpy as np
from PIL import Image

from common import DATA

CACHE = DATA / "gibs"
CACHE.mkdir(exist_ok=True)
TRUECOLOR = "MODIS_Terra_CorrectedReflectance_TrueColor"
BASE = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=EPSG:4326"
        "&FORMAT=image/jpeg&LAYERS={layers}&BBOX={s},{w},{n},{e}&WIDTH={W}&HEIGHT={H}&TIME={d}")


def snapshot(bbox, date, layers=TRUECOLOR, width=900):
    """bbox = (south, west, north, east) in degrees; returns an RGB array (EPSG:4326, so x = lon, y = lat linear)."""
    s, w, n, e = bbox
    h = int(width * (n - s) / (e - w))
    url = BASE.format(layers=layers, s=s, w=w, n=n, e=e, W=width, H=h, d=date)
    f = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".jpg")
    if not f.exists():
        for k in range(4):
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    f.write_bytes(r.read())
                break
            except Exception:  # noqa: BLE001
                if k == 3:
                    raise
    return np.array(Image.open(f).convert("RGB"))
