## ZZ-net for essential matrix estimation on the Reichstag dataset
The code in this repo is heavily using code from the [CNe](https://github.com/vcg-uvic/learned-correspondence-release) and [OANet](https://github.com/zjhthu/OANet) codebases.

### Dependencies
See the file `zz-net.yml` for dependencies. We used a conda environment inside a singularity container to run our experiments.

### Data
Obtain the Reichstag data from the CNe-repo and follow their instructions for
processing it. Then copy the reichstag data into a folder called `cne_datasets` inside this folder.

To obtain rotated test versions, run `python rotate_test_set.py`, which will generate rotated test sets under `cne_datasets/reichstag_rot30`, `cne_datasets/reichstag_rot60` and `cne_datasets/reichstag_rot180`.

### Training
Run `python ess_train.py`.

### Testing
Run `python run_test.py`.
