# Exploiting Review Sentiment to Smooth the Importance of Neighbors in Graph Collaborative Filtering

This is the official implementation of the paper _Exploiting Review Sentiment to Smooth the Importance of Neighbors in Graph
Collaborative Filtering_, under review as full paper at RecSys 2022.

This repository is heavily dependent on the framework **Elliot**, so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

We report the codes for the baselines, and the proposed model EGCF. In the following, we indicate the specific path for each model, and its reference backend (i.e., NumPy, TensorFlow or PyTorch):

- MostPop ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/elliot/recommender/unpersonalized/most_popular), backend: `NumPy`)
- BPRMF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/elliot/recommender/latent_factor_models/BPRMF), backend: `NumPy`)
- MultiVAE ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/elliot/recommender/autoencoders/vae), backend: `TensorFlow`)
- ConvMF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/convmf), backend: `TensorFlow`)
- DeepCoNN ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/deepconn), backend: `TensorFlow`)
- NGCF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/ngcf), backend: `PyTorch`)
- LightGCN ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/lightgcn), backend: `PyTorch`)
- SentiGCF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/egcf), backend: `PyTorch`)

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
At `./data/` you may find the files for the datasets, i.e., training, validation, and test sets, and the interactions file (for EGCF). In order not to overload this repository, we provide the links to Google Drive for the review-based side information:

| Dataset           | Link                                                                                        |
|-------------------|---------------------------------------------------------------------------------------------|
| **Baby**          | [drive](https://drive.google.com/file/d/1XKU7ZglJVvKimLPklexbTgiqrqU6WLnv/view?usp=sharing) |
| **Boys \& Girls** | [drive](https://drive.google.com/file/d/1X_2Sfqba7_3iSYYTeEYlCC12sQpcPdAD/view?usp=sharing) |
| **Men**           | [drive](https://drive.google.com/file/d/1bk8uHWBVOGkUmQjCzMEXX6BKa4UDtIW-/view?usp=sharing) |

After having downloaded the three zip files, just put them into `./data/amazon_baby/`, `./data/amazon_boys_girls/`, and `./data/amazon_men/`, respectively. 

Finally, run the bash scripts `./data/<dataset_name>/create_<dataset_name>.sh` to complete the procedure. Now you are all set, and you can start training and testing the models.

### Training and testing models
To train and evaluate models an all considered metrics, you may run the following command:

```
$ python -u start_experiments.py --model <model-name> --dataset <dataset-name>
```

where `<model-name>` and `<dataset-name>` refer to the name of the model to be run and the dataset on which to run the experiment, respectively.

The following table reports pointers to all configuration files. Please, use the same naming scheme as the row and the column headers for the model and dataset names.

|              |                                                            Baby                                                             |                                                       Boys<br/>&<br/>Girls                                                        |                                                            Men                                                             |
|--------------|:---------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------:|
| **MostPop**  | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/mostpop/amazon_baby.yml)  | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/mostpop/amazon_boys_girls.yml)  | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/mostpop/amazon_men.yml)  |
| **BPRMF**    |  [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/bprmf/amazon_baby.yml)   |  [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/bprmf/amazon_boys_girls.yml)   |  [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/bprmf/amazon_men.yml)   |
| **MultiVae** | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/multivae/amazon_baby.yml) | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/multivae/amazon_boys_girls.yml) | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/multivae/amazon_men.yml) |
| **ConvMF**   |  [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/convmf/amazon_baby.yml)  |  [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/convmf/amazon_boys_girls.yml)  |  [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/convmf/amazon_men.yml)  |
| **DeepCoNN** | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/deepconn/amazon_baby.yml) | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/deepconn/amazon_boys_girls.yml) | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/deepconn/amazon_men.yml) |
| **NGCF**     |   [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/ngcf/amazon_baby.yml)   |   [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/ngcf/amazon_boys_girls.yml)   |   [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/ngcf/amazon_men.yml)   |
| **LightGCN** | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/lightgcn/amazon_baby.yml) | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/lightgcn/amazon_boys_girls.yml) | [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/lightgcn/amazon_men.yml) |
| **EGCF**     |   [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/egcf/amazon_baby.yml)   |   [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/egcf/amazon_boys_girls.yml)   |   [config](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/blob/master/config_files/egcf/amazon_men.yml)   |

Results about calculated metrics are available in the folder `./results/<dataset-name>/performance/`. Specifically, you need to access the tsv file having the following name pattern: `rec_cutoff_<cutoff>_relthreshold_0_<datetime-experiment-end>.tsv`.
