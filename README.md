# FieldEncoders.jl
Field encoding for Neural Radiance Fields (NeRF) in Julia. The implementation is based on [`Field Encoders`](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/field_components/encodings.py) in [`nerfstudio`](https://github.com/nerfstudio-project/nerfstudio/tree/main).

## 1. NeRF Encoder
```bash
$ julia scripts/visualize_nerfencoder.jl
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

## 2. Hash Encoder
```bash
$ julia scripts/visualize_hashencoder.jl
```
### Input
<img src="./scripts/figures/hashencoder_input.png" width="200">

### Encoding 
<img src="./scripts/figures/hashencoder_encoded.png" width="1000">

## 3. Spherical Harmonic Encoder
```bash
$ julia scripts/visualize_sphericalharmonicencoder.jl
```
### Level 1
<img src="./scripts/figures/shencoder_encoded_level_1.png" width="100">

### Level 2
<img src="./scripts/figures/shencoder_encoded_level_2.png" width="300">

### Level 3
<img src="./scripts/figures/shencoder_encoded_level_3.png" width="500">

### Level 4
<img src="./scripts/figures/shencoder_encoded_level_4.png" width="700">

## Reference
- Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Terrance Wang, Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David Mcallister, Justin Kerr, and Angjoo Kanazawa. 2023. Nerfstudio: A Modular Framework for Neural Radiance Field Development. In ACM SIGGRAPH 2023 Conference Proceedings (SIGGRAPH '23). Association for Computing Machinery, New York, NY, USA, Article 72, 1–12.