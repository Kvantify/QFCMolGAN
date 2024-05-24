import argparse

from typing import Sequence


def get_GAN_config(args: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Quantum circuit configuration
    parser.add_argument(
        "--generator_type",
        type=str,
        default="classical",
        help="choice of generator",
        choices=["classical", "qubitvqc_hybrid", "qfcvqc_hybrid"],
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help="number of repeated variational quantum layer",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=8,
        help="number of qubits and dimension of domain labels",
    )
    parser.add_argument(
        "--update_qc",
        type=bool,
        default=True,
        help="choose to update the quantum circuit",
    )
    parser.add_argument("--qc_lr", type=float, default=None, help="learning rate of quantum circuit")
    parser.add_argument(
        "--qc_pretrained",
        type=bool,
        default=False,
        help="choose to use pretrained quantum circuit",
    )

    # Model configuration
    parser.add_argument(
        "--complexity", type=str, default="nr", help="dimension of domain labels", choices=["nr", "mr", "hr"]
    )
    parser.add_argument("--z_dim", type=int, default=8, help="dimension of domain labels")
    parser.add_argument(
        "--g_conv_dim",
        default=[128, 256, 512],
        help="number of conv filters in the first layer of G",
    )

    parser.add_argument(
        "--d_graph_conv_dims",
        type=int,
        default=[64, 32],  # NOTE: before [128, 64]
        help="number of conv filters in the first layer of D",
    )
    parser.add_argument(
        "--d_aux_dim",
        type=int,
        default=128,
        help="Size of MLPs in the aggregation part of D",
    )
    parser.add_argument(
        "--d_mlp_dims",
        type=int,
        default=[128, 64],
        help="number of nodes in the final MLP of D",
    )

    parser.add_argument(
        "--lambda_wgan",
        type=float,
        default=1.0,
        help="weight between RL and GAN. 1.0 for Pure GAN and 0.0 for Pure RL",
    )
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="weight for gradient penalty")
    parser.add_argument(
        "--post_method",
        type=str,
        default="softmax",
        choices=["softmax", "soft_gumbel", "hard_gumbel"],
    )

    parser.add_argument(
        "--reward_metric", type=str, default="sas,qed,unique", help="metric to use in the reward network"
    )

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=128, help="mini-batch size")
    parser.add_argument(
        "--pretraining_epochs",
        type=int,
        default=0,
        help="amount of epochs to pre-train the reward network using only wgan",
    )
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs for training D")
    parser.add_argument("--g_lr", type=float, default=0.001, help="learning rate for G")
    parser.add_argument("--d_lr", type=float, default=0.001, help="learning rate for D")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--n_critic", type=int, default=5, help="number of X updates per each Y update")
    parser.add_argument(
        "--critic_type",
        type=str,
        default="D",
        help="D for X=D and Y=G, G for X=G and Y=D",
    )
    parser.add_argument("--resume_epoch", type=int, default=None, help="resume training from this step")
    parser.add_argument(
        "--decay_every_epoch",
        type=int,
        default=None,
        help="decay learning rate by gamma every # epoch",
    )
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate decay rate")

    # Test configuration
    parser.add_argument("--test_epoch", type=int, default=None, help="test model from this step")
    parser.add_argument("--test_sample_size", type=int, default=None, help="number of testing molecules")

    # Miscellaneous
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    # Dataset directory
    parser.add_argument("--mol_data_dir", type=str, default="data/gdb9_9nodes.sparsedataset")

    # Saving directory
    parser.add_argument("--saving_dir", type=str, default="results/GAN/")

    # Step size
    parser.add_argument("--model_save_step", type=int, default=1)

    # Tensorboard
    parser.add_argument("--use_external_logger", action="store_true")

    config = parser.parse_args(args)

    return config
