## OLMoE

This repository provides an overview of all resources for the paper "OLMoE: ...".

### Artifacts

Links to all artifacts

### Pretraining

1. Clone [OLMo](https://github.com/allenai/OLMo) & create an environment with its dependencies via `cd OLMo; pip install -e .`
2. Run `pip install git+https://github.com/Muennighoff/megablocks.git@olmoe`
3. Setup a config file, there are a ton of config files in `configs/` (TODO: Add these)
4. Download the data from TODO & adapt the paths in your config file as needed
5. Submit your job. We used `bash olmoe.sh` using [beaker gantry](https://github.com/allenai/beaker-gantry) but you will likely need to change the script to work with your setup.

### Adaptation

TODO: Jacob / Nathan

### Evaluation

#### During pretraining

Evaluation during pretraining is done via the config files that run the pretraining.

#### After pretraining

TODO: Oyvindt / David

#### After adaptation

TODO: Jacob / Nathan

### Visuals

All plots are here: https://colab.research.google.com/drive/15PTwmoxcbrwWKG6ErY44hlJlLLKAj7Hx?usp=sharing

### Citation
