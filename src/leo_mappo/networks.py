from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Adam:
    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.m = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.t = 0

    def step(self, gradients: dict[str, np.ndarray], max_grad_norm: float) -> None:
        if not all(np.all(np.isfinite(gradient)) for gradient in gradients.values()):
            raise FloatingPointError("optimizer received a non-finite gradient")
        self.t += 1
        norm = np.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients.values()))
        scale = min(1.0, max_grad_norm / max(norm, 1e-8))
        for name, parameter in self.parameters.items():
            gradient = gradients[name] * scale
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * gradient**2
            m_hat = self.m[name] / (1.0 - 0.9**self.t)
            v_hat = self.v[name] / (1.0 - 0.999**self.t)
            parameter -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)


@dataclass
class MLP:
    input_dim: int
    hidden_dims: int | list[int] | tuple[int, ...]
    output_dim: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        if isinstance(self.hidden_dims, int):
            self.hidden_dims = (self.hidden_dims,)
        else:
            self.hidden_dims = tuple(int(dim) for dim in self.hidden_dims)
        if not self.hidden_dims or any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("hidden_dims must contain at least one positive dimension")
        dims = (self.input_dim, *self.hidden_dims, self.output_dim)
        self.parameters: dict[str, np.ndarray] = {}
        for layer in range(len(dims) - 1):
            scale = np.sqrt(1.0 / dims[layer]) if layer < len(dims) - 2 else 0.01
            self.parameters[f"w{layer + 1}"] = self.rng.normal(
                0.0, scale, (dims[layer], dims[layer + 1])
            )
            self.parameters[f"b{layer + 1}"] = np.zeros(dims[layer + 1], dtype=np.float64)

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, tuple[list[np.ndarray], list[np.ndarray]]]:
        activations = [inputs]
        preactivations: list[np.ndarray] = []
        current = inputs
        layer_count = len(self.hidden_dims) + 1
        for layer in range(1, layer_count + 1):
            linear = current @ self.parameters[f"w{layer}"] + self.parameters[f"b{layer}"]
            preactivations.append(linear)
            current = np.tanh(linear) if layer < layer_count else linear
            activations.append(current)
        return current, (activations, preactivations)

    def backward(self, grad_outputs: np.ndarray, cache: tuple[list[np.ndarray], list[np.ndarray]]) -> dict[str, np.ndarray]:
        gradients, _ = self.backward_with_input(grad_outputs, cache)
        return gradients

    def backward_with_input(
        self,
        grad_outputs: np.ndarray,
        cache: tuple[list[np.ndarray], list[np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        activations, _ = cache
        layer_count = len(self.hidden_dims) + 1
        gradients: dict[str, np.ndarray] = {}
        grad = grad_outputs
        for layer in range(layer_count, 0, -1):
            gradients[f"w{layer}"] = activations[layer - 1].T @ grad
            gradients[f"b{layer}"] = grad.sum(axis=0)
            if layer > 1:
                grad = (grad @ self.parameters[f"w{layer}"].T) * (1.0 - activations[layer - 1] ** 2)
        return gradients, grad @ self.parameters["w1"].T


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class GRU:
    input_dim: int
    hidden_dim: int
    rng: np.random.Generator

    def __post_init__(self) -> None:
        input_scale = np.sqrt(1.0 / self.input_dim)
        hidden_scale = np.sqrt(1.0 / self.hidden_dim)
        self.parameters: dict[str, np.ndarray] = {}
        for gate in ("z", "r", "h"):
            self.parameters[f"w{gate}"] = self.rng.normal(
                0.0, input_scale, (self.input_dim, self.hidden_dim)
            )
            self.parameters[f"u{gate}"] = self.rng.normal(
                0.0, hidden_scale, (self.hidden_dim, self.hidden_dim)
            )
            self.parameters[f"b{gate}"] = np.zeros(self.hidden_dim, dtype=np.float64)

    def forward_step(
        self, inputs: np.ndarray, hidden_before: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
        z_gate = _sigmoid(
            inputs @ self.parameters["wz"]
            + hidden_before @ self.parameters["uz"]
            + self.parameters["bz"]
        )
        r_gate = _sigmoid(
            inputs @ self.parameters["wr"]
            + hidden_before @ self.parameters["ur"]
            + self.parameters["br"]
        )
        candidate = np.tanh(
            inputs @ self.parameters["wh"]
            + (r_gate * hidden_before) @ self.parameters["uh"]
            + self.parameters["bh"]
        )
        hidden = (1.0 - z_gate) * hidden_before + z_gate * candidate
        if not np.all(np.isfinite(hidden)):
            raise FloatingPointError("GRU produced a non-finite hidden state")
        return hidden, (inputs, hidden_before, z_gate, r_gate, candidate)

    def forward_sequence(
        self, inputs: np.ndarray, initial_hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, ...]]]:
        if inputs.ndim != 3:
            raise ValueError("GRU inputs must have shape [sequence_length, batch, input_dim]")
        hidden = initial_hidden
        sequence = []
        cache = []
        for step_inputs in inputs:
            hidden, step_cache = self.forward_step(step_inputs, hidden)
            sequence.append(hidden)
            cache.append(step_cache)
        return np.stack(sequence, axis=0), hidden, cache

    def backward_sequence(
        self,
        grad_hidden_sequence: np.ndarray,
        cache: list[tuple[np.ndarray, ...]],
        grad_final_hidden: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        grad_inputs = np.zeros(
            (len(cache), cache[0][0].shape[0], self.input_dim), dtype=np.float64
        )
        grad_next = (
            np.zeros_like(cache[0][1]) if grad_final_hidden is None else grad_final_hidden.copy()
        )
        for step in range(len(cache) - 1, -1, -1):
            inputs, hidden_before, z_gate, r_gate, candidate = cache[step]
            grad_hidden = grad_hidden_sequence[step] + grad_next
            grad_candidate = grad_hidden * z_gate
            grad_z = grad_hidden * (candidate - hidden_before)
            grad_before = grad_hidden * (1.0 - z_gate)

            grad_candidate_pre = grad_candidate * (1.0 - candidate**2)
            reset_hidden = r_gate * hidden_before
            gradients["wh"] += inputs.T @ grad_candidate_pre
            gradients["uh"] += reset_hidden.T @ grad_candidate_pre
            gradients["bh"] += grad_candidate_pre.sum(axis=0)
            grad_inputs[step] += grad_candidate_pre @ self.parameters["wh"].T
            grad_reset_hidden = grad_candidate_pre @ self.parameters["uh"].T
            grad_r = grad_reset_hidden * hidden_before
            grad_before += grad_reset_hidden * r_gate

            grad_r_pre = grad_r * r_gate * (1.0 - r_gate)
            gradients["wr"] += inputs.T @ grad_r_pre
            gradients["ur"] += hidden_before.T @ grad_r_pre
            gradients["br"] += grad_r_pre.sum(axis=0)
            grad_inputs[step] += grad_r_pre @ self.parameters["wr"].T
            grad_before += grad_r_pre @ self.parameters["ur"].T

            grad_z_pre = grad_z * z_gate * (1.0 - z_gate)
            gradients["wz"] += inputs.T @ grad_z_pre
            gradients["uz"] += hidden_before.T @ grad_z_pre
            gradients["bz"] += grad_z_pre.sum(axis=0)
            grad_inputs[step] += grad_z_pre @ self.parameters["wz"].T
            grad_before += grad_z_pre @ self.parameters["uz"].T
            grad_next = grad_before
        return gradients, grad_inputs, grad_next


def masked_softmax(logits: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if np.any(np.sum(masks > 0.0, axis=-1) == 0):
        raise ValueError("action mask contains a row with no feasible action")
    masked_logits = np.where(masks > 0.0, logits, -1e9)
    shifted = masked_logits - masked_logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted) * masks
    return exponentials / np.maximum(exponentials.sum(axis=-1, keepdims=True), 1e-12)
