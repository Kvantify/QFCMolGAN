from typing import Optional, Type

import torch
import torch.nn as nn

from models.layers import GraphConvolution, GraphAggregation, MultiDenseLayers
from models.q_generator import (
    QubitInputGen,
    QFCInputGen,
    QFCPhaseEncodingStrategy,
    EquallySpacedRepetitions,
)


# decoder_adj in MolGAN/models/__init__.py in original TF implementation
# Implementation-MolGAN-PyTorch/models_gan.py Generator
class ClassicalGenerator(nn.Module):
    """Generator network of MolGAN"""

    def __init__(
        self,
        conv_dims: list[int],
        z_dim: int,
        vertexes: int,
        edges: int,
        nodes: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv_dims = conv_dims
        self.z_dim = z_dim
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        self.multi_dense_layers = MultiDenseLayers(z_dim, conv_dims, self.activation_f)
        self.edges_layer = nn.Linear(conv_dims[-1], edges * vertexes * vertexes)
        self.nodes_layer = nn.Linear(conv_dims[-1], vertexes * nodes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output: torch.Tensor = self.multi_dense_layers(x)

        edges_logits: torch.Tensor = self.edges_layer(output).view(-1, self.edges, self.vertexes, self.vertexes)
        edges_logits = (edges_logits + edges_logits.permute(0, 1, 3, 2)) / 2
        edges_logits = self.dropout(edges_logits.permute(0, 2, 3, 1))

        nodes_logits: torch.Tensor = self.nodes_layer(output)
        nodes_logits = self.dropout(nodes_logits.view(-1, self.vertexes, self.nodes))

        return edges_logits, nodes_logits


class QubitVqcHybridGenerator(nn.Module):
    """Hybrid quantum-classical generator with a qubit-based VQC for generating a prior distribution"""

    def __init__(
        self,
        conv_dims: list[int],
        vertexes: int,
        edges: int,
        nodes: int,
        dropout_rate: float,
        n_qubits: int,
        n_layers: int,
    ) -> None:
        super().__init__()

        self.classical_generator = ClassicalGenerator(
            conv_dims,
            n_qubits,
            vertexes,
            edges,
            nodes,
            dropout_rate,
        )

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = 2

        self.vqc = QubitInputGen(n_layers, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ignores the input and generates random input from a VQC instead"""

        batch_size = x.size(0)
        z = torch.rand((batch_size, self.input_dim), dtype=torch.float32, device="cpu")

        input = self.vqc(z).to(x.device)

        return self.classical_generator(input)


class QfcVqcHybridGenerator(nn.Module):
    """Hybrid quantum-classical generator with a qubit-based VQC for generating a prior distribution"""

    def __init__(
        self,
        conv_dims: list[int],
        vertexes: int,
        edges: int,
        nodes: int,
        dropout_rate: float,
        n_states: int,
        n_layers: int,
        n_buffer_states: int = 0,
        input_dim: int = 2,
        encoding_strategy: Optional[Type[QFCPhaseEncodingStrategy]] = None,
    ) -> None:
        super().__init__()

        self.classical_generator = ClassicalGenerator(
            conv_dims,
            n_states,
            vertexes,
            edges,
            nodes,
            dropout_rate,
        )

        self.n_states = n_states
        self.n_buffers = n_buffer_states
        self.n_layers = n_layers
        self.input_dim = input_dim

        if encoding_strategy is None:
            self.encoding_strategy: Type[QFCPhaseEncodingStrategy] = EquallySpacedRepetitions
        else:
            self.encoding_strategy = encoding_strategy

        self.vqc = QFCInputGen(n_layers, n_states, n_buffer_states, self.encoding_strategy, self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ignores the input and generates random input from a VQC instead"""

        batch_size = x.size(0)
        z = torch.rand((batch_size, self.input_dim), dtype=torch.float32)
        z = z.to(x.device)

        input = self.vqc(z)

        return self.classical_generator(input)


# From Implementation-MolGAN-PyTorch/models_gan.py Discriminator
class Discriminator(nn.Module):
    """Discriminator network of MolGAN"""

    def __init__(
        self,
        graph_conv_dims: list[int],
        aux_dim: int,
        mlp_dims: list[int],
        m_dim: int,
        b_dim: int,
        with_features: bool = False,
        f_dim: int = 0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.graph_conv_dims = graph_conv_dims
        self.aux_dim = aux_dim
        self.mlp_dims = mlp_dims
        self.m_dim = m_dim
        self.b_dim = b_dim
        self.with_features = with_features
        self.f_dim = f_dim
        self.dropout_rate = dropout_rate

        self.activation_f = nn.Tanh()
        # discriminator
        self.gcn_layer = GraphConvolution(m_dim, graph_conv_dims, b_dim, with_features, f_dim, dropout_rate)
        self.agg_layer = GraphAggregation(
            graph_conv_dims[-1] + m_dim,
            aux_dim,
            self.activation_f,
            with_features,
            f_dim,
            dropout_rate,
        )
        self.multi_dense_layers = MultiDenseLayers(aux_dim, mlp_dims, self.activation_f, dropout_rate)
        self.output_layer = nn.Linear(mlp_dims[-1], 1)

    def forward(
        self,
        adjacency_tensor: torch.Tensor,
        hidden: Optional[torch.Tensor],
        node: torch.Tensor,
        activation: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        adj = adjacency_tensor[:, :, :, 1:].permute(0, 3, 1, 2)
        adj = torch.nan_to_num(adj)
        h = self.gcn_layer(node, adj, hidden)
        h = self.agg_layer(node, h, hidden)
        # h = self.agg_layer(h, node, hidden)
        h = self.multi_dense_layers(h)

        output = self.output_layer(h)
        output = activation(output) if activation is not None else output

        return output, h
