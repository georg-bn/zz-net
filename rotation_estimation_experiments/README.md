## Rotation estimation from noisy correspondences

### Installation

The code is written using [PyTorch Lightning](https://www.pytorchlightning.ai/), a pytorch package for streamlining
machine learning using pytorch. You will need that installed to make the code run. Pytorch Lightning is
open source, and can be installed using `pip`.

    pip install pytorch-lightning

### Generate data

We provide the data we used for training and testing in the directories `data/test_data_paper` and `data/train_valid_data_paper`.
You can also choose to generate your own data. To use the same parameters as in the paper, simply run

    python generate_point_cloud_data.py paper

You can also specify your own point cloud size, train, valid and test sizes, outlier ratios and inlier noise level through 

    python generate_point_cloud_data.py name outlier_ratio cloud_size train_size valid_size test_size outlier_noise

Files ``name_ratio.pt`` with test and validation data and ``name_ratio_test.pt`` will appear in the data directory ready to be
used for experiments.

### Training models

We provide code for the four models (broad ZZ, deep ZZ, aCNe- and PointNet) used in the experiments. To rerun the experiments
from the paper on the data (with the same random seed) with outlier ratio `ratio`, use the commands

    python ZZNet.py broad ratio paper
    python ZZNet.py deep ratio paper
    python aCNE.py ratio paper
    python pointNet.py ratio paper

We can also customize the training data, number of epochs, learning rate and random seed. If you used `mydata` as the name
when generating the data, run e.g.

    python ZZNet.py broad mydata.pt epochs learning_rate seed

and similar for the other models. PyTorch Lightning will automatically produce logfiles in the `logs` directory - these can be viewed
with tensorboard. The trained models will be saved as `trained_model_modelname_dataname.pt` in the `trained_models` directory

### Testing 

To evaluate a model, use ``evaluation_rotation_net.py``. We provide our pretrained models in the ``trained_models/pretrained`` directory -
any models you train will end up in the ``trained_models`` directory. To rerun the tests on the pretrained models on the
test data with outlier ratio `ratio` used in the paper, run

    python evaluate_rotation_net.py paper ratio type

where `type` specifies the type of model to be evaluated:

 * ``zb`` for broad ZZ
 * ``zd`` for deep ZZ
 * ``a`` for aCNe-
 * ``p`` for pointNet

To test a model '``my_model.pt`` located the ``trained_models``-directory on data ``my_data.pt`` in the ``data`` directory, run

    python evaluate_rotation_net.py my_model.pt my_data.pt type

Note that if the test data is generated as above, it will have a name of the form `name_ratio_test.pt`.