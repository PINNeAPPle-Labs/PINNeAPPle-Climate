# runs e solver arquivados

Em 2026-10-29 os diretórios `runs/` e `solver/` desta pasta foram substituídos por
arquivos `.tar.zst` verificados, em `~/Documents/GitHub/.arquivos-climate/`:

| diretório | original | arquivado | arquivos conferidos |
|---|---|---|---|
| `03_bumper_beam_transolver/runs`   | 9,30 GB | 2,58 GB | 21.412 |
| `03_bumper_beam_transolver/solver` | 0,50 GB | 0,26 GB |  2.549 |

Os `.dvc` correspondentes (`runs.dvc`, `solver.dvc`) continuam no lugar, com os hashes
originais, e os dados **estão no remote DVC** (`ssh://pinneapple-arena/srv/dados/pinneapple/climate`).
Conferido em 2026-10-30: `dvc status -c` responde "Cache and remote 'servidor' are in sync" e
os objetos existem em `files/md5/` no servidor.

Ou seja, há duas formas de recuperar, e a mais simples é a do DVC:

```bash
cd ~/Documents/GitHub/pinneapple-labs/PINNeAPPle-Climate
dvc pull experiments/03_bumper_beam_transolver/runs.dvc
```

`data/` e `01_sst_satellite_forecasting/data/` **não** foram arquivados: ali o tar+zstd
não ganhava nada (1,40 GB -> 1,34 GB e 0,17 GB -> 0,17 GB), então os originais ficaram
como estavam. `hazards_forecasting/` também ficou intacto porque tem 43 arquivos
rastreados pelo git.

## Restaurar

```bash
cd ~/Documents/GitHub/pinneapple-labs/PINNeAPPle-Climate/experiments/03_bumper_beam_transolver
for f in ~/Documents/GitHub/.arquivos-climate/bumper_*.tar.zst; do
  tar --use-compress-program=unzstd -xf "$f"
done
```

## Listar sem extrair

```bash
tar --use-compress-program=unzstd -tf ~/Documents/GitHub/.arquivos-climate/bumper_runs.tar.zst | head
```
