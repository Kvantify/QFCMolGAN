from typing import Sequence, Optional, Type

import numpy as np
import pennylane as qml  # type: ignore
import torch
import torch.nn as nn
from scipy import linalg as la  # type: ignore


# Enable CUDA device if available
torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

V_EOM = 9  # strength of the EOM driving amplitude. Realistic up to ~10


class QFCPhaseEncodingStrategy(nn.Module):
    def __init__(self, n_states: int, n_buffers: int, input_dim: int) -> None:
        super().__init__()
        self.n_states = n_states
        if n_buffers > 0:
            self.n_buffers: Optional[int] = n_buffers
            self.n_total = self.n_states + 2 * self.n_buffers
        else:
            self.n_buffers = None
            self.n_total = n_states

        if input_dim < 1:
            raise Exception("Input dimension must be at least 1.")  # Unless sampling from a distribution??
        self.input_dim = input_dim


class EquallySpacedRepetitions(QFCPhaseEncodingStrategy):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2:
            raise Exception
        if input.shape[1] != self.input_dim:
            raise Exception("Input dimension 1 does not match expected.")
        if self.input_dim < 2:
            raise NotImplementedError

        output = torch.zeros((input.shape[0], self.n_states))

        num_repetitions = np.ceil(self.n_total / self.input_dim).item()

        # Perform the repetition
        repeated_phase = input.repeat(1, int(num_repetitions))
        output[
            ...,
            self.n_buffers : -self.n_buffers if self.n_buffers is not None else None,
        ] = repeated_phase[..., : self.n_states]

        return output.to(input.device).type(torch.cfloat)


class Learnable(QFCPhaseEncodingStrategy):
    def __init__(self, n_states: int, n_buffers: int, input_dim: int) -> None:
        super().__init__(n_states, n_buffers, input_dim)
        self.mat = nn.Parameter(torch.rand((self.n_states, self.input_dim)))
        self.buffer_states = torch.zeros(self.n_buffers).unsqueeze(0) if self.n_buffers is not None else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.ndim != 2:
            raise Exception
        if input.shape[1] != self.input_dim:
            raise Exception("Input dimension 1 does not match expected.")

        out = torch.matmul(self.mat, input)
        return (
            (torch.cat((self.buffer_states, out, self.buffer_states), dim=1) if self.buffer_states is not None else out)
            .to(input.device)
            .type(torch.cfloat)
        )


class QFCInputGen(nn.Module):
    """Quantum QFC-based VQC input distribution generator for a hybrid quantum-classical generator"""

    def __init__(
        self,
        n_layers: int,
        n_equivalent_qubits: int,
        n_buffers: int,
        encoding_strategy: Type[QFCPhaseEncodingStrategy],
        input_dim: int,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.n_output_size = n_equivalent_qubits
        self.n_states = 2**n_equivalent_qubits
        self.n_buffers = n_buffers
        self.n_total = self.n_states + 2 * n_buffers

        F = la.dft(self.n_total, scale="sqrtn")
        phases = np.exp(-1j * V_EOM * np.cos(2 * np.pi / self.n_total * np.arange(self.n_total)))
        D = np.diag(phases)
        eom = torch.tensor(np.matmul(F, np.matmul(D, F.conj().T)), dtype=torch.complex64)
        self.register_buffer("eom", eom, persistent=False)

        self.encoding_strategy = encoding_strategy(self.n_states, n_buffers, input_dim)
        self.ps_params = nn.Parameter(torch.rand(n_layers, self.n_total, dtype=torch.cfloat))

    def equivalent_qubit_measurement(self, state: torch.Tensor) -> torch.Tensor:
        probs = torch.abs(state) ** 2
        probs /= torch.sum(probs, dim=1, keepdim=True)
        indices = torch.arange(probs.size(1), device=state.device)
        vals = torch.zeros(probs.size(0), self.n_output_size, device=state.device)
        for i in range(self.n_output_size):
            n_states = 2 ** (i + 1)
            filter = (indices % n_states) > (n_states - 1) / 2
            # torch.index_select(probs, 1, indices[filter])
            vals[:, i] = probs[:, filter].sum(dim=1)

        return vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (n_batch, 2)

        output: shape (n_batch, n_states)
        """

        phases = self.encoding_strategy(x)
        state: torch.Tensor = torch.exp(-1j * phases) / np.sqrt(self.n_total)

        for j in range(self.n_layers):
            state *= torch.exp(-1j * self.ps_params[j]).unsqueeze(0)  # PS operation
            state = torch.matmul(self.eom, state.T).T  # EOPM operation

        return self.equivalent_qubit_measurement(state)


class QubitInputGen(nn.Module):
    """Quantum qubit-based VQC input distribution generator for a hybrid quantum-classical generator"""

    def __init__(self, n_layers: int, n_qubits: int):
        super().__init__()
        self.n_layers = n_layers
        self.n_qubits = n_qubits

        input_dev = qml.device("default.qubit", wires=n_qubits)  # TODO: Very slow! Maybe change to Qiskit instead?
        self.params = nn.Parameter(torch.rand(n_layers, 2 * n_qubits - 1) * 2 * np.pi - np.pi)

        @qml.qnode(input_dev, interface="torch", diff_method="backprop")
        def input_gen_circuit(inputs: torch.Tensor, w: Sequence) -> Sequence:
            for i in range(n_qubits):
                qml.RY(inputs[0], wires=i)
                qml.RZ(inputs[1], wires=i)

            for j in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(w[j][i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CRZ(w[j][i + n_qubits], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = input_gen_circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: shape (n_batch, 2)

        output: shape (n_batch, n_qubits)
        """
        params = torch.clone(self.params).to("cpu")
        outs = [torch.tensor(self.circuit(i, params), dtype=torch.float32) for i in inputs]
        return torch.stack(outs)
