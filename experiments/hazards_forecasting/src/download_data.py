"""Download the public catalogs used by the hazard-forecasting experiments (idempotent).

IBTrACS v04r01 (best tracks built from geostationary/polar satellite fixes, Dvorak, microwave): data/ibtracs/
USGS ComCat M>=4.5, 1980-2025:                                                                data/usgs/
NOAA NCEI/WDS tsunami events and run-ups:                                                      data/tsunami/
Smithsonian GVP Holocene eruptions (WFS):                                                      data/volcano/
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
IBTRACS = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
           "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv")


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("retry", k, url[:90], e, file=sys.stderr)
            time.sleep(3 * (k + 1))
    raise RuntimeError(url)


def ibtracs():
    out = DATA / "ibtracs" / "ibtracs.ALL.csv"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(get(IBTRACS))


def usgs(y0=1980, y1=2025):
    d = DATA / "usgs"
    d.mkdir(parents=True, exist_ok=True)
    for y in range(y0, y1 + 1):
        f = d / f"comcat_{y}.csv"
        if f.exists():
            continue
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&orderby=time-asc&minmagnitude=4.5"
               f"&starttime={y}-01-01&endtime={y + 1}-01-01")
        f.write_bytes(get(url))


def tsunami():
    d = DATA / "tsunami"
    d.mkdir(parents=True, exist_ok=True)
    for kind in ("events", "runups"):
        f = d / f"{kind}.json"
        if f.exists():
            continue
        items, page = [], 1
        while True:
            r = json.loads(get(f"https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/{kind}"
                               f"?minYear=1900&maxYear=2026&page={page}"))
            items += r["items"]
            if page >= r["totalPages"]:
                break
            page += 1
        f.write_text(json.dumps(items))


def volcano():
    f = DATA / "volcano" / "gvp_eruptions.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(get("https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs?service=WFS&version=1.0.0"
                          "&request=GetFeature&typeName=GVP-VOTW:Smithsonian_VOTW_Holocene_Eruptions"
                          "&outputFormat=application/json"))


if __name__ == "__main__":
    for fn in (ibtracs, usgs, tsunami, volcano):
        fn()
        print("ok", fn.__name__)
