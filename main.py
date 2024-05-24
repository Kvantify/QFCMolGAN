import os
import logging

from argparse import Namespace

from rdkit import RDLogger  # type: ignore
from torch.backends import cudnn

from utils.args import get_GAN_config
from utils.utils_io import get_date_postfix

from solver import Solver

# Remove flooding logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def main(config: Namespace) -> None:
    # For fast training
    cudnn.benchmark = True

    # Timestamp
    if config.mode == "train":
        a_train_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir, a_train_time)
        config.log_dir_path = os.path.join(config.saving_dir, config.mode, "log_dir")
        config.model_dir_path = os.path.join(config.saving_dir, config.mode, "model_dir")
        config.img_dir_path = os.path.join(config.saving_dir, config.mode, "img_dir")
    else:
        a_test_time = get_date_postfix()
        config.saving_dir = os.path.join(config.saving_dir)
        config.log_dir_path = os.path.join(config.saving_dir, "post_test", a_test_time, "log_dir")
        config.model_dir_path = os.path.join(config.saving_dir, "model_dir")
        config.img_dir_path = os.path.join(config.saving_dir, "post_test", a_test_time, "img_dir")

    # Create directories if not exist
    if not os.path.exists(config.log_dir_path):
        os.makedirs(config.log_dir_path)
    if not os.path.exists(config.model_dir_path):
        os.makedirs(config.model_dir_path)
    if not os.path.exists(config.img_dir_path):
        os.makedirs(config.img_dir_path)

    # Logger
    if config.mode == "train":
        log_p_name = os.path.join(config.log_dir_path, a_train_time + "_logger.log")
        from importlib import reload

        reload(logging)
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)
    elif config.mode == "test":
        log_p_name = os.path.join(config.log_dir_path, a_test_time + "_logger.log")
        from importlib import reload

        reload(logging)
        logging.basicConfig(filename=log_p_name, level=logging.INFO)
        logging.info(config)
    else:
        raise NotImplementedError

    # Solver for training and test MolGAN
    if config.mode == "train":
        solver = Solver(config, logging)
    elif config.mode == "test":
        solver = Solver(config, logging)
    else:
        raise NotImplementedError

    solver.train_and_validate()


if __name__ == "__main__":
    config = get_GAN_config([])

    # # Logging
    config.use_external_logger = True
    config.saving_dir = r"results/GAN"

    # # Dataset
    # molecule dataset dir
    # config.mol_data_dir = r'data/gdb9_9nodes.sparsedataset'
    config.mol_data_dir = r"data/qm9_5k.sparsedataset"

    # # Quantum
    # quantum circuit to generate inputs of MolGAN
    config.generator_type = "qfcvqc_hybrid"  # one of: "classical", "qubitvqc_hybrid", "qfcvqc_hybrid"
    # number of qubit of quantum circuit
    config.qubits = 4
    # number of layer of quantum circuit
    config.layer = 4
    # TODO: Enable turning off update of the parameters of quantum circuit
    # config.update_qc = False

    # # Training
    config.mode = "train"
    # the complexity of generator
    config.complexity = "mr"
    # metric(s) used in the reward network for RL loss
    config.reward_metric = "sas,logp,qed"
    # batch size
    config.batch_size = 32
    # input noise dimension
    config.z_dim = 32  # NOTE: previously 8
    # start epoch (<= 0)
    config.pretraining_epochs = 50
    # number of epoch
    config.num_epochs = 250
    # n_critic
    config.n_critic = 2
    # critic type
    config.critic_type = "D"
    # 1.0 for pure WGAN and 0.0 for pure RL
    config.lambda_wgan = 0.01
    # weight decay
    config.decay_every_epoch = 60
    config.gamma = 0.1

    # # Testing
    # config.mode = "test"
    # config.complexity = "mr"
    # config.test_sample_size = 10
    # config.z_dim = 8
    # config.test_epoch = 100

    # See QumolGAN article
    if config.complexity == "nr":  # no reduction
        config.g_conv_dim = [128, 256, 512]
    elif config.complexity == "mr":  # medium reduction
        config.g_conv_dim = [128]
    elif config.complexity == "hr":  # high reduction
        config.g_conv_dim = [16]
    else:
        raise ValueError("Please enter an valid model complexity from 'mr', 'hr' or 'nr'!")

    print(config)

    main(config)
