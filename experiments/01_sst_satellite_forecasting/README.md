# 01 — Previsão de temperatura da superfície do mar a partir de satélite

Experimento de clima único que permanece neste repositório (ver `experiments/MOVED.md`).

- `src/`: código do experimento (`sst_experiment.py`) e do download dos dados (`download_oisst.py`).
- `paper/`: artigo (`make_paper.py` e o PDF gerado).
- `figures/`: figuras do artigo.
- `results/`: métricas e logs da execução final (`results.json`, `final_training.json`, `hpo.json`, `log.txt`).

Não versionados por tamanho (ficam só na máquina local): `data/` (OISST, ~178 MB), os pesos `*.pt` de `results/` e `results_smoke/`, e a pasta `results_smoke/` completa.
Para reproduzir, rode `src/download_oisst.py` e depois `src/sst_experiment.py`.
