"""Skill of the adaptive ensemble by lead time (1-4 weeks) for the five volcanoes, vs persistence and vs climatology."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGS, RESULTS

d = json.loads((RESULTS / "volcano_viirs.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
for k, (key, ttl) in enumerate((("skill_vs_naive", "vs persistence (last week's value)"), ("skill_vs_climatology", "vs climatology"))):
    for n, v in d.items():
        est = [v[f"h{h}"][key][0] for h in range(1, 5)]
        lo = [v[f"h{h}"][key][1] for h in range(1, 5)]
        hi = [v[f"h{h}"][key][2] for h in range(1, 5)]
        off = list(d).index(n) * 0.06
        ax[k].errorbar([h + off for h in range(1, 5)], est, yerr=[[e - l for e, l in zip(est, lo)], [u - e for e, u in zip(est, hi)]],
                       marker="o", capsize=2, label=n)
    ax[k].axhline(0, color="grey", lw=0.8)
    ax[k].set(xlabel="lead (weeks)", ylabel="skill", title=f"VIIRS thermal series, skill {ttl}", xticks=[1, 2, 3, 4])
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "volcano_skill_by_lead.png", dpi=120)
