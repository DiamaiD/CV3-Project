# CV3-Project

Physics video world model: CNN-VAE + flow-matching DiT, trained on 64×64 simulated environments.

**Full write up (results, graphs, gifs):** https://diamaid.github.io/CV3-Project/

## Run

- Training GUI: `python -m src.gui`
- Data generation GUI: `python -m src.datagen_gui`
- Command line: `python -m src.main --env_name <env> --data_dir data/<dataset>` (all training options: `python -m src.main --help`)
