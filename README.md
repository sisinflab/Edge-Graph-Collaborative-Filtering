# Edge Graph Collaborative Filtering

This is the official implementation of the paper _Edge Graph Collaborative Filtering_, under review as short paper at SIGIR 2022.

This repository is heavily dependent on the framework **Elliot**, so we suggest you refer to the official GitHub [page](https://github.com/sisinflab/elliot) and [documentation](https://elliot.readthedocs.io/en/latest/).

We report the codes for the baselines, and the proposed model EGCF. In the following, we indicate the specific path for each model, and its reference backend (i.e., NumPy, TensorFlow or PyTorch):

- MostPop ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/elliot/recommender/unpersonalized/most_popular), backend: `NumPy`)
- BPRMF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/elliot/recommender/latent_factor_models/BPRMF), backend: `NumPy`)
- MultiVAE ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/elliot/recommender/autoencoders/vae), backend: `TensorFlow`)
- ConvMF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/convmf), backend: `TensorFlow`)
- DeepCoNN ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/deepconn), backend: `TensorFlow`)
- NGCF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/ngcf), backend: `PyTorch`)
- LightGCN ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/lightgcn), backend: `PyTorch`)
- EGCF ([path](https://github.com/sisinflab/Edge-Graph-Collaborative-Filtering/tree/master/external/models/egcf), backend: `PyTorch`)

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

to be continued...

### Datasets
At `./data/` you may find the files for the datasets, i.e., training, validation, and test sets, and the interactions file (for EGCF). In order not to overload this repository, we provide the links to Google Drive for the review-based side information:

| Dataset           | Link                                                                                        |
|-------------------|---------------------------------------------------------------------------------------------|
| **Baby**          | [drive](https://drive.google.com/file/d/1XKU7ZglJVvKimLPklexbTgiqrqU6WLnv/view?usp=sharing) |
| **Boys \& Girls** | [drive](https://drive.google.com/file/d/1X_2Sfqba7_3iSYYTeEYlCC12sQpcPdAD/view?usp=sharing) |
| **Men**           | [drive](https://drive.google.com/file/d/1bk8uHWBVOGkUmQjCzMEXX6BKa4UDtIW-/view?usp=sharing) |

After having downloaded the three zip files, just put them into `./data/amazon_baby/`, `./data/amazon_boys_girls`, and `./data/amazon_men`, respectively. 

Finally, run the bash scripts `./data/<dataset_name>/create_<dataset_name>.sh` to complete the procedure. Now you are all set, and you can start training and testing the models.