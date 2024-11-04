
<div style="text-align: center;">
    <img src="/static/images/cosed.png" alt="CoSeD" width="250" height="250">
</div>

# Contrastive Sequential-Diffusion Learning: Non-linear and Multi-Scene Instructional Video Synthesis

[![arXiv](https://img.shields.io/badge/arXiv-2407.11814-b31b1b.svg)](https://arxiv.org/abs/2407.11814)
[![Project](https://img.shields.io/badge/Project-Website-9cf.svg)](https://novasearch.github.io/CoSeD/)



This is the official repository for Contrastive Sequential-Diffusion Learning: Non-linear and Multi-Scene Instructional Video Synthesis (WACV 2025).

## Code Structure

`generate_only_latents.py` - script for generating images using latents and performing inference with the SoftAttention model

`latents_singleton.py` - contains the `Latents` class, a singleton for managing latent vectors

`sequence_predictor.py` - contains the `SoftAttention` model and related functions for processing text and image embeddings

`videos_softattention.py` - script for generating videos from images using a diffusion pipeline and concatenating them into a single video

## Citation
If you find CoSeD useful for your research and applications, please cite using this BibTeX:
```
@misc{ramos2024contrastivesequentialdiffusionlearningnonlinear,
      title={Contrastive Sequential-Diffusion Learning: Non-linear and Multi-Scene Instructional Video Synthesis},
      author={Vasco Ramos and Yonatan Bitton and Michal Yarom and Idan Szpektor and Joao Magalhaes},
      year={2024},
      eprint={2407.11814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11814},
}
```
