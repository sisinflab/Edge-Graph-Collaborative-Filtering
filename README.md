# Edge Graph Collaborative Filtering

This is the official implementation of the paper _Edge Graph Collaborative Filtering_, under review as short paper at SIGIR 2022.

This repository is heavily dependent on the framework **Elliot**, so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

We report the codes for the baselines, and the proposed model EGCF. In the following, we indicate the specific path for each model, and its reference backend (i.e., NumPy, TensorFlow or PyTorch):

- MostPop (path: `./elliot/recommender/unpersonalized/most_popular/`, backend: `NumPy`)
- BPRMF (path: `./elliot/recommender/latent_factor_models/BPRMF/`, backend: `NumPy`)
- MultiVAE (path: `./elliot/recommender/autoencoders/vae/`, backend: `TensorFlow`)
- ConvMF (path: `./external/models/convmf/`, backend: `TensorFlow`)
- DeepCoNN (path: `./external/models/deepconn/`, backend: `TensorFlow`)
- NGCF (path: `./external/models/ngcf/`, backend: `PyTorch`)
- LightGCN (path: `./external/models/lightgcn/`, backend: `PyTorch`)
- EGCF (path: `./external/models/egcf/`, backend: `PyTorch`)

As for `TensorFlow`, we tested our models using the gpu version `2.3.2`, with CUDA `10.1` and cuDNN `7.6`. 

As for `PyTorch`, we tested our models using the version `1.10.2`, with CUDA `10.2` and cuDNN `8.0`. Additionally, graph-based models require `PyTorch Geometric`, which is compatible with the versions of CUDA and `PyTorch` we indicated above.

### Installation guidelines: scenario #1
If you have the possibility to install two different versions of CUDA on your workstation (i.e.,`10.1` and `10.2`), you may create two different virtual environments with the requirements files we included in the repository, as follows:

```
# TENSORFLOW ENVIRONMENT (CUDA 10.1, cuDNN 7.6)

$ python3 -m venv venv_tf
$ source venv_tf/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements_tf.txt
```

```
# PYTORCH ENVIRONMENT (CUDA 10.2, cuDNN 8.0)

$ python3 -m venv venv_pt
$ source venv_pt/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements_pt.txt
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
```

### Installation guidelines: scenario #2
A more convenient way of running experiments is to instantiate two docker containers having CUDA `10.1` and CUDA `10.2` already installed. We provide the Dockerfile to build each of the two containers.

to be continued...