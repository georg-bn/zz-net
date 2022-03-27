import argparse
import torch
from pathlib import Path
from oan_test import test
import ess_data
import essential_matrix_model


def run_all(config):
    for checkpoint in Path(config.log_dir).rglob("checkpoints/*.ckpt"):
        print(f"Checkpoint: {checkpoint}")
        model = essential_matrix_model.EssMatrixTensorNet()

        config.save_file_best = checkpoint
        for r in [0, 30, 60, 180]:
            config.rotate = r
            res_path = Path(config.res_path_base) / \
                f"rot{r}"
            res_path.mkdir(exist_ok=True, parents=True)
            config.res_path = str(res_path)
            va_res = run_reichstag_test(model, config)
            print(f"rotate {r} degrees")
            print(va_res)


def run_reichstag_test(model, config):
    """The main function."""

    if config.rotate == 0:
        test_dataset = ess_data.EssentialDatasetPkl(
            "reichstag", "test", data_folder=config.test_data_folder)
    else:
        print(f"Using rotated test data: reichstag_rot{config.rotate}")
        test_dataset = ess_data.EssentialDatasetPkl(
            f"reichstag_rot{config.rotate}",
            "test", data_folder=config.test_data_folder)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=False,
        collate_fn=ess_data.collate_fn)

    return test(test_loader, model, config)


def str2bool(v):
    return v.lower() in ("true", "1")


if __name__ == "__main__":

    # ----------------------------------------
    # Define configuration
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_folder", type=str, default="cne_datasets", help=""
        "path for loading data")
    parser.add_argument(
        "--log_dir", type=str, default="tb_logs/std_exp1/version_0", help=""
        "base path for tensorboard logs (where checkpoints are saved)")
    parser.add_argument(
        "--res_path_base", type=str, default="test_results", help=""
        "base path for saving results")
    # The following are arguments from OANet
    # which we don't change from the below defaults:
    parser.add_argument(
        "--use_ransac", type=str2bool, default=False, help=""
        "use ransac when testing?")
    parser.add_argument(
        "--obj_top_k", type=int, default=-1, help=""
        "number of keypoints above the threshold to use for "
        "essential matrix estimation. put -1 to use all. ")
    parser.add_argument(
        "--use_fundamental", type=str2bool, default=False, help=""
        "train fundamental matrix estimation. Default: False")
    parser.add_argument(
        "--loss_classif", type=float, default=1.0, help=""
        "weight of the classification loss")
    parser.add_argument(
        "--loss_essential", type=float, default=0.5, help=""
        "weight of the essential loss")
    parser.add_argument(
        "--loss_essential_init_iter", type=int, default=20000, help=""
        "initial iterations to run only the classification loss")
    parser.add_argument(
        "--geo_loss_margin", type=float, default=0.1, help=""
        "clamping margin in geometry loss")
    parser.add_argument(
        "--obj_geod_th", type=float, default=1e-4, help=""
        "theshold for the good geodesic distance")

    config, unparsed = parser.parse_known_args()
    run_all(config)
