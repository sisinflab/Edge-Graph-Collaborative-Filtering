# Reshaping Graph Recommendation with Edge Graph Collaborative Filtering and Customer Reviews

This is the official implementation of the paper _Reshaping Graph Recommendation with Edge Graph Collaborative Filtering and Customer Reviews_, accepted as full paper at DL4SR@CIKM 2022.

This repository is heavily dependent on the framework **Elliot**, so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

We report the codes for the baselines, and the proposed model EGCF. In the following, we indicate the specific path for each model, and its reference backend (i.e., NumPy, TensorFlow or PyTorch):

- MostPop ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/elliot/recommender/unpersonalized/most_popular/most_popular.py), backend: `NumPy`)
- BPRMF ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/elliot/recommender/latent_factor_models/BPRMF/BPRMF.py), backend: `NumPy`)
- MultiVAE ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/elliot/recommender/autoencoders/vae/multi_vae.py), backend: `TensorFlow`)
- ConvMF ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/convmf/ConvMF.py), backend: `TensorFlow`)
- RMG ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/rmg/RMG.py), backend: `TensorFlow`)
- NGCF ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/ngcf/NGCF.py), backend: `PyTorch`)
- LightGCN ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/lightgcn/LightGCN.py), backend: `PyTorch`)
- GAT ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/gat/GAT.py), backend: `PyTorch`)
- DGCF ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/dgcf/DGCF.py), backend: `PyTorch`)
- EGCF ([path](https://anonymous.4open.science/r/Edge-Graph-Collaborative-Filtering-D0D3/external/models/egcf/EGCF.py), backend: `PyTorch`)

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
A more convenient way of running experiments is to instantiate two docker containers having CUDA `10.1` and CUDA `10.2` already installed, respectively. We provide the Dockerfile to build each of the two containers.

Make sure you have Docker and NVIDIA Container Toolkit installed on your machine (you may refer to this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)).

Then, you may use the following Docker images to instantiate two containers equipped with CUDA `10.1` and CUDA `10.2`, respectively:

- Container Docker with CUDA `10.1` and cuDNN `7.6` (the environment for `TensorFlow`): [link](https://hub.docker.com/layers/nvidia/cuda/10.1-cudnn7-devel-ubuntu18.04/images/sha256-c38db79d18f576fa84b041638b2d560cd7d450791279a5cdfc044fb5708e431b?context=explore)
- Container Docker with CUDA `10.2` and cuDNN `8.0` (the environment for `PyTorch`): [link](https://hub.docker.com/layers/nvidia/cuda/10.2-cudnn8-devel-ubuntu18.04/images/sha256-3d1aefa978b106e8cbe50743bba8c4ddadacf13fe3165dd67a35e4d904f3aabe?context=explore)

After the setup of your Docker containers, you may follow the exact same guidelines as scenario #1.
### Datasets
At `./data/` you may find all tsv files for the datasets, i.e., training, validation, and test sets, and the interactions file (for EGCF). In order not to overload this repository, we provide the links to Google Drive for the review-based side information (both for EGCF and other baselines):

| Dataset           | Link EGCF                                                                                   | Link Review Baselines |
|-------------------|---------------------------------------------------------------------------------------------|-----------------------|
| **Baby**          | [drive](https://drive.google.com/file/d/1XKU7ZglJVvKimLPklexbTgiqrqU6WLnv/view?usp=sharing) |[drive](https://drive.google.com/file/d/11wDeIZqWA4VHnJJF5qAbEIJnPhsJ850j/view?usp=sharing) |
| **Boys \& Girls** | [drive](https://drive.google.com/file/d/1X_2Sfqba7_3iSYYTeEYlCC12sQpcPdAD/view?usp=sharing) |[drive](https://drive.google.com/file/d/1jFC5WMxlQW7nUOmadXTJ9ZREW2lFSdZh/view?usp=sharing) |
| **Men**           | [drive](https://drive.google.com/file/d/1bk8uHWBVOGkUmQjCzMEXX6BKa4UDtIW-/view?usp=sharing) |[drive](https://drive.google.com/file/d/1BhSJf2ZptrvB96TQRszfGQF0lGF8oWKb/view?usp=sharing) |

After having downloaded the six zip files, just put them into `./data/amazon_baby/`, `./data/amazon_boys_girls/`, and `./data/amazon_men/`, respectively. 

Finally, run the bash scripts `./data/<dataset_name>/create_<dataset_name>.sh` to complete the procedure. Now you are all set, and you can start training and testing the models.

### Training and testing models
To train and evaluate models an all considered metrics, you may run the following command:

```
$ python -u start_experiments.py --model <model-name> --dataset <dataset-name>
```

where `<model-name>` and `<dataset-name>` refer to the name of the model to be run and the dataset on which to run the experiment, respectively.

You may find all configutation files at `./config_files/<model-name>/<dataset-name>.yml`, where all hyperparameter spaces and the exploration strategies are reported.

As for EGCF (i.e., our proposed model), configuration files follow the pattern `./config_files/egcf/<dataset-name>_<hop_number>.yml`, useful to run the study on the hop number and select the best model configuration.

Results about calculated metrics are available in the folder `./results/<dataset-name>/performance/`. Specifically, you need to access the tsv file having the following name pattern: `rec_cutoff_<cutoff>_relthreshold_0_<datetime-experiment-end>.tsv`.
