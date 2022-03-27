import argparse
import ess_data
import essential_matrix_model
import pytorch_lightning as pl
import torch


def train_standard(train_loader,
                   val_loader,
                   nbr_epochs,
                   experiment_name,
                   log_folder,
                   nbr_gpus):
    """
    Train an EssMatrixTensorNet on the EssentialDataset train_loader,
    validate on val_loader.
    """
    model = essential_matrix_model.EssMatrixTensorNet()
    logger = pl.loggers.TensorBoardLogger(log_folder, name=experiment_name)
    trainer = pl.Trainer(gpus=nbr_gpus, max_epochs=nbr_epochs, logger=logger)
    trainer.fit(model, train_loader, val_loader)


def run_experiment(args):
    train_dataset = ess_data.EssentialDatasetPkl(
        "reichstag", "train", data_folder=args.datapath)

    val_dataset = ess_data.EssentialDatasetPkl(
        "reichstag", "validation", data_folder=args.datapath)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=16,
        pin_memory=False,
        collate_fn=(ess_data.collate_fn))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=False,
        collate_fn=(ess_data.collate_fn))
    train_standard(train_loader,
                   val_loader,
                   nbr_epochs=1,
                   experiment_name=args.exp_name,
                   log_folder=args.logpath,
                   nbr_gpus=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logpath',
        dest='logpath',
        type=str,
        default="tb_logs")
    parser.add_argument(
        '--datapath',
        dest='datapath',
        type=str,
        default="cne_datasets")
    parser.add_argument(
        '--exp_name',
        dest='exp_name',
        type=str,
        default="std_exp1")
    args = parser.parse_args()
    run_experiment(args)
