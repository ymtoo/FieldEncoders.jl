# FieldEncoders.jl

## 1. NeRF Encoder
```bash
$ julia --project scripts/visualize_nerfencoder.jl
```
### Input
<img src="./scripts/figures/nerfencoder_input.png" width="200">

### Encoding 
<img src="./scripts/figures/nerfencoder_encoded.png" width="1000">

### Encoding with 0.01 covariance magnitude
<img src="./scripts/figures/nerfencoder_encoded_0.01_cov.png" width="1000">

### Encoding with 0.1 covariance magnitude
<img src="./scripts/figures/nerfencoder_encoded_0.1_cov.png" width="1000">

### Encoding with 1.0 covariance magnitude
<img src="./scripts/figures/nerfencoder_encoded_1.0_cov.png" width="1000">

## Reference
- Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David Mcallister, Justin Kerr, and Angjoo Kanazawa. 2023. Nerfstudio: A Modular Framework for Neural Radiance Field Development. In ACM SIGGRAPH 2023 Conference Proceedings (SIGGRAPH '23). Association for Computing Machinery, New York, NY, USA, Article 72, 1–12.