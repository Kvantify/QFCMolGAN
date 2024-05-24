import datetime
import os
import time
from argparse import Namespace
from collections import defaultdict
from typing import Any

from frechetdist import frdist  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray

from data.sparse_molecular_dataset import SparseMolecularDataset
from models.models import (
    ClassicalGenerator,
    Discriminator,
    QubitVqcHybridGenerator,
    QfcVqcHybridGenerator,
)
from utils.external_logger import WandbLogger
from utils.utils import MolecularMetrics, all_scores, save_mol_img


def upper(m: torch.Tensor, a: torch.Tensor):
    res = torch.zeros((m.shape[0], 36, 5)).to(m.device).long()
    for i in range(m.shape[0]):
        for j in range(5):
            tmp_m = m[i, :, :, j]
            idx = torch.triu_indices(9, 9, offset=1)

            res[i, :, j] = tmp_m[list(idx)]
    res = torch.cat((res, a), dim=1)
    return res


def wasserstein_loss(y_true, y_pred):
    return torch.mean(y_true * y_pred)


class Solver(object):
    """Solver for training and testing MolGAN"""

    def __init__(self, config: Namespace, log: Any = None) -> None:
        """Initialize configurations"""

        # Log
        self.log = log

        # loss
        self.wasserstein = True  # True only for C dis
        # Data loader
        self.data = SparseMolecularDataset()
        self.data.load(config.mol_data_dir)

        # Quantum
        self.generator_type = config.generator_type
        if config.generator_type == "qubit_patch":
            self.q_patch_gen_config = config.q_patch_gen_config
        self.n_layers = config.layer
        self.n_qubits = config.qubits
        self.n_states = config.qubits
        self.qc_pretrained = config.qc_pretrained

        # Model configurations
        self.z_dim = config.z_dim
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.g_conv_dim = config.g_conv_dim
        self.d_graph_conv_dims = config.d_graph_conv_dims
        self.d_aux_dim = config.d_aux_dim
        self.d_mlp_dims = config.d_mlp_dims
        self.la = config.lambda_wgan
        self.la_gp = config.lambda_gp
        self.post_method = config.post_method

        # RL reward suggested by medicinal chemist
        self.metric = config.reward_metric

        # Training configurations
        self.batch_size = config.batch_size
        # Start epoch - usually 0, but negative if a pretraining with WGAN before RL is desired
        assert config.pretraining_epochs > 0
        self.start_epoch = -max(config.pretraining_epochs, 0)
        self.num_epochs = config.num_epochs + config.pretraining_epochs
        # number of steps per epoch
        self.num_steps = len(self.data) // self.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        # learning rate decay
        self.gamma = config.gamma
        self.decay_every_epoch = config.decay_every_epoch

        # # critic
        self.n_critic = config.n_critic
        self.critic_type = config.critic_type

        # Training or test
        self.mode = config.mode
        self.resume_epoch = config.resume_epoch

        # Testing configurations
        self.test_epoch = config.test_epoch
        self.test_sample_size = config.test_sample_size

        # Tensorboard
        self.use_external_logger = config.use_external_logger
        if self.mode == "train" and config.use_external_logger:
            self.logger = WandbLogger("QFC_MolGAN", config=config.__dict__)

        # GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device, flush=True)

        # Directories
        self.log_dir_path = config.log_dir_path
        self.model_dir_path = config.model_dir_path
        self.img_dir_path = config.img_dir_path

        # Step size to save the model
        self.model_save_step = config.model_save_step

        # Build the model
        self.build_model()

    def build_model(self) -> None:
        """Create a generator, a discriminator and a v net"""

        # Models
        match self.generator_type:
            case "qfcvqc_hybrid":
                self.G: torch.nn.Module = QfcVqcHybridGenerator(
                    self.g_conv_dim,
                    self.data.vertexes,
                    self.data.bond_num_types,
                    self.data.atom_num_types,
                    self.dropout,
                    self.n_states,
                    self.n_layers,
                )

            case "qubitvqc_hybrid":
                self.G = QubitVqcHybridGenerator(
                    self.g_conv_dim,
                    self.data.vertexes,
                    self.data.bond_num_types,
                    self.data.atom_num_types,
                    self.dropout,
                    self.n_qubits,
                    self.n_layers,
                )

            case "classical":
                # TODO: Add classical noise generator
                self.G = ClassicalGenerator(
                    self.g_conv_dim,
                    self.z_dim,
                    self.data.vertexes,
                    self.data.bond_num_types,
                    self.data.atom_num_types,
                    self.dropout,
                )
            case _:
                raise ValueError(f"generator_type {self.generator_type} not recoqnized.")

        self.D = Discriminator(
            self.d_graph_conv_dims,
            self.d_aux_dim,
            self.d_mlp_dims,
            self.m_dim,
            self.b_dim - 1,
            self.dropout,
        )
        self.V = Discriminator(
            self.d_graph_conv_dims,
            self.d_aux_dim,
            self.d_mlp_dims,
            self.m_dim,
            self.b_dim - 1,
            self.dropout,
        )

        # compile - unfortunately, not compatible with gradient penalties yet
        # torch.compile(G)
        # torch.compile(D)
        # torch.compile(V)

        # # Optimizers can be RMSprop or Adam
        # self.g_optimizer = torch.optim.RMSprop(self.G.parameters(), self.g_lr)
        self.g_optimizer = torch.optim.AdamW(self.G.parameters(), self.g_lr)

        # self.d_optimizer = torch.optim.RMSprop(self.D.parameters(), self.d_lr)
        # self.d_optimizer = torch.optim.SGD(self.D.parameters(), lr=0.2)
        self.d_optimizer = torch.optim.AdamW(self.D.parameters(), self.d_lr)

        # self.v_optimizer = torch.optim.RMSprop(self.V.parameters(), self.g_lr)
        self.v_optimizer = torch.optim.AdamW(self.V.parameters(), self.g_lr)

        # Watch models in logger
        # self.logger.watch_model(self.G)
        # self.logger.watch_model(self.D)
        # self.logger.watch_model(self.V)

        # Print the networks
        self.print_network(self.G, "G", self.log)
        self.print_network(self.D, "D", self.log)
        self.print_network(self.V, "V", self.log)

        # Bring the network to GPU
        self.G.to(self.device)
        self.D.to(self.device)
        self.V.to(self.device)

    @staticmethod
    def print_network(model: torch.nn.Module, name: str, log: Any = None) -> None:
        """Print out the network information"""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        if log is not None:
            log.info(model)
            log.info(name)
            log.info("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters: int) -> None:
        """Restore the trained generator and discriminator"""
        print("Loading the trained models from step {}...".format(resume_iters))
        G_path = os.path.join(self.model_dir_path, f"{resume_iters}-G.ckpt")
        D_path = os.path.join(self.model_dir_path, f"{resume_iters}-D.ckpt")
        V_path = os.path.join(self.model_dir_path, f"{resume_iters}-V.ckpt")
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))

    def update_lr(self, gamma: float) -> None:
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.d_optimizer.param_groups:
            param_group["lr"] *= gamma
        for param_group in self.g_optimizer.param_groups:
            param_group["lr"] *= gamma

    def reset_grad(self) -> None:
        """Reset the gradient buffers"""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.v_optimizer.zero_grad()

    def gradient_penalty(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def label2onehot(self, labels: torch.Tensor, dim: int) -> torch.Tensor:
        """Convert label indices to one-hot vectors"""
        out = torch.zeros(list(labels.size()) + [dim]).to(self.device)
        out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.0)
        return out

    def sample_z(self, batch_size: int) -> np.ndarray:
        """Sample the random noise"""
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    @staticmethod
    def postprocess(
        inputs: tuple[torch.Tensor, torch.Tensor], method: str, temperature: float = 1.0
    ) -> list[torch.Tensor]:
        """Convert the probability matrices into label matrices"""

        def listify(x):
            return x if isinstance(x, (list, tuple)) else [x]

        def delistify(x):
            return x if len(x) > 1 else x[0]

        if method == "soft_gumbel":
            softmax = [
                F.gumbel_softmax(
                    e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                    hard=False,
                ).view(e_logits.size())
                for e_logits in listify(inputs)
            ]
        elif method == "hard_gumbel":
            softmax = [
                F.gumbel_softmax(
                    e_logits.contiguous().view(-1, e_logits.size(-1)) / temperature,
                    hard=True,
                ).view(e_logits.size())
                for e_logits in listify(inputs)
            ]
        else:
            softmax = [F.softmax(e_logits / temperature, -1) for e_logits in listify(inputs)]

        return [delistify(e) for e in (softmax)]

    def reward(self, mols: list[Any]) -> np.ndarray:
        """Calculate the rewards of mols"""
        rr: ArrayLike = 1.0
        for m in ("logp,sas,qed,unique" if self.metric == "all" else self.metric).split(","):
            if m == "np":
                rr *= MolecularMetrics.natural_product_scores(mols, norm=True)
            elif m == "logp":
                rr *= MolecularMetrics.water_octanol_partition_coefficient_scores(mols, norm=True)
            elif m == "sas":
                rr *= MolecularMetrics.synthetic_accessibility_score_scores(mols, norm=True)
            elif m == "qed":
                rr *= MolecularMetrics.quantitative_estimation_druglikeness_scores(mols, norm=True)
            elif m == "novelty":
                rr *= MolecularMetrics.novel_scores(mols, self.data)
            elif m == "dc":
                rr *= MolecularMetrics.drugcandidate_scores(mols, self.data)
            elif m == "unique":
                rr *= MolecularMetrics.unique_scores(mols)
            elif m == "diversity":
                rr *= MolecularMetrics.diversity_scores(mols, self.data)
            elif m == "validity":
                rr *= MolecularMetrics.valid_scores(mols)
            else:
                raise ValueError(f"{m} is not defined as a metric")
        return rr.reshape(-1, 1)  # type: ignore

    def train_and_validate(self) -> None:
        """Train and validate function"""
        self.start_time = time.time()

        # start training from scratch or resume training
        start_epoch = self.start_epoch
        end_epoch = start_epoch + self.num_epochs
        if self.resume_epoch is not None and self.mode == "train":
            start_epoch = self.resume_epoch
            self.restore_model(self.resume_epoch)
        # restore models for test
        elif self.test_epoch is not None and self.mode == "test":
            self.restore_model(self.test_epoch)

        # TODO: Option for only loading quantum circuit??

        else:
            print("Training From Scratch...")

        # start training loop or test phase
        if self.mode == "train":
            print("Start training...")
            for i in range(start_epoch, end_epoch):
                self.train_or_valid(epoch_i=i, train_val_test="train")
                self.train_or_valid(epoch_i=i, train_val_test="val")
        elif self.mode == "test":
            print("Start testing...")
            assert self.resume_epoch is not None or self.test_epoch is not None
            self.train_or_valid(epoch_i=start_epoch, train_val_test="val")
        else:
            raise ValueError(f"mode not recognized: {self.mode}")

    def get_gen_mols(self, n_hat: torch.Tensor, e_hat: torch.Tensor, method: str) -> NDArray:
        """Convert edges and nodes matrices into molecules"""
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = (
            torch.max(edges_hard, -1)[1],
            torch.max(nodes_hard, -1)[1],
        )
        mols = [
            self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
            for e_, n_ in zip(edges_hard, nodes_hard)
        ]
        return np.array(mols)

    def get_reward(self, n_hat: torch.Tensor, e_hat: torch.Tensor, method: str) -> torch.Tensor:
        """Get the reward from edges and nodes matrices"""
        (edges_hard, nodes_hard) = self.postprocess((e_hat, n_hat), method)
        edges_hard, nodes_hard = (
            torch.max(edges_hard, -1)[1],
            torch.max(nodes_hard, -1)[1],
        )
        mols = [
            self.data.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True)
            for e_, n_ in zip(edges_hard, nodes_hard)
        ]
        reward = torch.from_numpy(self.reward(mols)).to(self.device)
        return reward

    def save_checkpoints(self, epoch_i: int) -> None:
        """store the models and quantum circuit"""
        G_path = os.path.join(self.model_dir_path, f"{epoch_i + 1}-G.ckpt")
        D_path = os.path.join(self.model_dir_path, f"{epoch_i + 1}-D.ckpt")
        V_path = os.path.join(self.model_dir_path, f"{epoch_i + 1}-V.ckpt")
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)
        torch.save(self.V.state_dict(), V_path)

        print("Saved model checkpoints into {}...".format(self.model_dir_path))
        if self.log is not None:
            self.log.info("Saved model checkpoints into {}...".format(self.model_dir_path))

    def train_or_valid(self, epoch_i: int, train_val_test: str = "val") -> None:
        """Train or valid function"""
        # The first several epochs using on WGAN Loss RL to help train V (reward network)
        if epoch_i < 0:
            cur_la = 1
        else:
            cur_la = self.la

        # Iterations
        if train_val_test == "val":
            if self.mode == "train":
                the_step = 1
                print("[Validating]")
            elif self.mode == "test":
                the_step = 1
                print("[Testing]")
            else:
                raise ValueError
        else:
            the_step = self.num_steps

        # Recordings
        losses = defaultdict(list)
        scores = defaultdict(list)

        # Training loop
        for a_step in range(the_step):
            # non-Quantum part
            if train_val_test == "val":
                mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch(self.test_sample_size)
                if self.test_sample_size is None:
                    z_array = self.sample_z(a.shape[0])
                else:
                    z_array = self.sample_z(self.test_sample_size)
            elif train_val_test == "train":
                mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                z_array = self.sample_z(self.batch_size)
            else:
                raise ValueError

            # ######### Preprocess input data ##########
            a_tensor = torch.from_numpy(a).to(self.device).long()  # adjacency
            x_tensor = torch.from_numpy(x).to(self.device).long()  # node
            a_tensor = self.label2onehot(a_tensor, self.b_dim)
            x_tensor = self.label2onehot(x_tensor, self.m_dim)

            # ax_tensor = upper(a_tensor, x_tensor)
            z = torch.from_numpy(z_array).to(self.device).float()

            # for external logging
            loss_metrics = {}

            # current steps
            cur_step = self.num_steps * (epoch_i - self.start_epoch) + a_step

            # ######### Train the discriminator ##########

            # compute loss with real inputs
            logits_real, features_real = self.D(a_tensor, None, x_tensor)  # // C dis

            # compute loss with fake inputs
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)  # // C dis

            # Compute losses for gradient penalty
            eps = torch.rand(logits_real.size(0), 1, 1, 1).to(self.device)
            x_int0 = (eps * a_tensor + (1.0 - eps) * edges_hat).requires_grad_(True)
            x_int1 = (eps.squeeze(-1) * x_tensor + (1.0 - eps.squeeze(-1)) * nodes_hat).requires_grad_(True)
            y, _ = self.D(x_int0, None, x_int1)
            grad_penalty = self.gradient_penalty(y, x_int0) + self.gradient_penalty(y, x_int1)

            # # ----- # TODO: Include configurable d_loss function --------
            # d_loss_real = d_loss(logits_real, target_real)
            # d_loss_fake = d_loss(logits_fake, target_fake)

            d_loss_real = -torch.mean(logits_real)
            d_loss_fake = torch.mean(logits_fake)
            loss_D = (
                torch.mean(d_loss_fake) + torch.mean(d_loss_real) + self.la_gp * grad_penalty
                if self.wasserstein
                else (d_loss_real + d_loss_fake)
            )  # modified for wasserstein

            if cur_la > 0:
                losses["D/loss_real"].append(d_loss_real.item())
                losses["D/loss_fake"].append(d_loss_fake.item())
                losses["D/loss_gp"].append(grad_penalty.item())
                losses["D/loss"].append(loss_D.item())

                # external_logging
                if train_val_test == "train":
                    loss_metrics["D/loss_real"] = d_loss_real.item()
                    loss_metrics["D/loss_fake"] = d_loss_fake.item()
                    loss_metrics["D/loss_gp"] = grad_penalty.item()
                    loss_metrics["D/loss"] = loss_D.item()
                elif train_val_test == "val":
                    loss_metrics["D/val_loss_real"] = d_loss_real.item()
                    loss_metrics["D/val_loss_fake"] = d_loss_fake.item()
                    loss_metrics["D/val_loss_gp"] = grad_penalty.item()
                    loss_metrics["D/val_loss"] = loss_D.item()

            # Optimise discriminator
            if train_val_test == "train" and cur_la > 0:
                if self.critic_type == "D":
                    # training D for n_critic-1 times followed by G one time
                    if (cur_step == 0) or (cur_step % self.n_critic != 0):
                        self.reset_grad()
                        loss_D.backward()
                        self.d_optimizer.step()

                else:
                    # training G for n_critic-1 times followed by D one time
                    if (cur_step != 0) and (cur_step % self.n_critic == 0):
                        self.reset_grad()
                        loss_D.backward()
                        self.d_optimizer.step()
                        print("-")

            # ######### Train the generator ##########

            # Z-to-target
            edges_logits, nodes_logits = self.G(z)
            # Postprocess with Gumbel softmax
            (edges_hat, nodes_hat) = self.postprocess((edges_logits, nodes_logits), self.post_method)

            logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)  # // C dis

            # Value losses (RL)
            value_logit_real, _ = self.V(a_tensor, None, x_tensor, torch.sigmoid)
            value_logit_fake, _ = self.V(edges_hat, None, nodes_hat, torch.sigmoid)

            # Real Reward
            reward_r = torch.from_numpy(self.reward(mols.tolist())).to(self.device)
            # Fake Reward
            reward_f = self.get_reward(nodes_hat, edges_hat, self.post_method)

            # Losses Update
            loss_G = -logits_fake

            # Loss function - either abs or squared diff
            # loss_V = (value_logit_real - reward_r) ** 2 + (value_logit_fake - reward_f) ** 2
            loss_V = torch.abs(value_logit_real - reward_r) + torch.abs(value_logit_fake - reward_f)

            loss_RL = -value_logit_fake  # trained reward
            # loss_RL = -reward_f  # exact reward

            loss_G = torch.mean(loss_G)
            loss_V = torch.mean(loss_V)
            loss_RL = torch.mean(loss_RL)
            losses["G/loss"].append(loss_G.item())
            losses["RL/loss"].append(loss_RL.item())
            losses["V/loss"].append(loss_V.item())

            # external_logging
            if train_val_test == "train":
                loss_metrics["G/loss"] = loss_G.item()
                loss_metrics["RL/loss"] = loss_RL.item()
                loss_metrics["V/loss"] = loss_V.item()
            elif train_val_test == "val":
                loss_metrics["G/val_loss"] = loss_G.item()
                loss_metrics["RL/val_loss"] = loss_RL.item()
                loss_metrics["V/val_loss"] = loss_V.item()

            print(
                "d_loss {:.2f} d_fake {:.2f} d_real {:.2f} g_loss: {:.2f}".format(
                    loss_D.item(), d_loss_fake.item(), d_loss_real.item(), loss_G.item()
                )
            )
            print(
                "======================= {} ==============================".format(datetime.datetime.now()),
                flush=True,
            )
            alpha = torch.abs(loss_G.detach() / loss_RL.detach()).detach()
            train_step_G = cur_la * loss_G + (1.0 - cur_la) * alpha * loss_RL

            train_step_V = loss_V

            # Optimise generator and reward network
            if train_val_test == "train":
                if self.critic_type == "D":
                    # training D for n_critic-1 times followed by G one time
                    if (cur_step != 0) and (cur_step % self.n_critic) == 0:
                        self.reset_grad()
                        if cur_la < 1.0 or epoch_i < 0:
                            train_step_G.backward(retain_graph=True)
                            train_step_V.backward()
                            self.g_optimizer.step()
                            self.v_optimizer.step()
                        else:
                            train_step_G.backward(retain_graph=True)
                            self.g_optimizer.step()
                else:
                    # training G for n_critic-1 times followed by D one time
                    if (cur_step == 0) or (cur_step % self.n_critic != 0):
                        self.reset_grad()
                        if cur_la < 1.0:
                            train_step_G.backward(retain_graph=True)
                            train_step_V.backward()
                            self.g_optimizer.step()
                            self.v_optimizer.step()
                        else:
                            train_step_G.backward(retain_graph=True)
                            self.g_optimizer.step()

            # ######### Frechet distribution ##########
            (edges_hard, nodes_hard) = self.postprocess((edges_logits, nodes_logits), "hard_gumbel")
            edges_hard, nodes_hard = (
                torch.max(edges_hard, -1)[1],
                torch.max(nodes_hard, -1)[1],
            )
            R = [list(a_i.reshape(-1)) for a_i in a]
            F = [list(edges_hard_i.reshape(-1).to("cpu")) for edges_hard_i in edges_hard]
            # F =  F.cpu()
            fd_bond = frdist(R, F)

            R = [list(x[i]) + list(a_i.reshape(-1)) for i, a_i in enumerate(a)]
            F = [list(nodes_hard[i].to("cpu")) + list(e_i.reshape(-1).to("cpu")) for i, e_i in enumerate(edges_hard)]
            fd_bond_atom = frdist(R, F)

            loss_metrics["FD/bond"] = fd_bond
            loss_metrics["FD/bond_atom"] = fd_bond_atom

            losses["FD/bond"].append(fd_bond)
            losses["FD/bond_atom"].append(fd_bond_atom)

            if self.use_external_logger and a_step % 10 == 0:
                if train_val_test == "train":
                    self.logger.log_metrics(loss_metrics, step=cur_step)
                else:
                    self.logger.log_metrics(loss_metrics)

            # ######### Miscellaneous ##########

            # Decay learning rates
            # if epoch_i > 0 and self.decay_every_epoch:
            #     if a_step == 0 and (epoch_i + 1) % self.decay_every_epoch == 0:
            #         self.update_lr(self.gamma)

            # Get scores
            # if train_val_test == 'val':
            if a_step % 10 == 0:
                mols = self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
                m0, m1 = all_scores(mols.tolist(), self.data, norm=True)  # 'mols' is output of Fake Reward
                for k1, v1 in m1.items():
                    scores[k1].append(v1)
                for k0, v0 in m0.items():
                    scores[k0].append(v0[np.nonzero(v0)].mean())

                # Save checkpoints
                if self.mode == "train":
                    if epoch_i > 0 and (epoch_i + 1) % self.model_save_step == 0:
                        self.save_checkpoints(epoch_i=epoch_i)

                    if self.use_external_logger:
                        for k0, v0 in m0.items():
                            self.logger.log_metrics({k0: v0[np.nonzero(v0)].mean()})
                        self.logger.log_metrics(m1)

                # Saving molecule images
                mol_f_name = os.path.join(self.img_dir_path, "mol-{}.png".format(epoch_i))
                save_mol_img(mols, mol_f_name, is_test=self.mode == "test")

                # Print out training information
                et = time.time() - self.start_time
                et_date = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]:".format(
                    et_date, (epoch_i - self.start_epoch) + 1, self.num_epochs
                )

                is_first = True
                for tag, value in losses.items():
                    if is_first:
                        log += f"\n{tag}: {np.mean(value):.2f}"
                        is_first = False
                    else:
                        log += f", {tag}: {np.mean(value):.2f}"
                is_first = True
                for tag, value in scores.items():
                    if is_first:
                        log += f"\n{tag}: {np.mean(value):.2f}"
                        is_first = False
                    else:
                        log += f", {tag}: {np.mean(value):.2f}"
                print(log)

                if self.log is not None:
                    self.log.info(log)
