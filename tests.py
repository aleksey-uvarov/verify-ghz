import unittest
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
import numpy as np

import parityoscillations


class TestParityOscillations(unittest.TestCase):

    def test_depolarizing_only_h(self):
        """The depolarizing channel is applied only after
        the first Hadamard gate. Everything else is clean."""
        my_noise_model = NoiseModel()
        p = 0.2
        single_qubit_error = depolarizing_error(p, 1)
        my_noise_model.add_all_qubit_quantum_error(single_qubit_error, ['h'])
        phi_steps = 40
        num_qubits = 5
        phi_values = np.linspace(0, 4 * 2 * np.pi / num_qubits, num=phi_steps)
        par_osc_fid, error_osc = parityoscillations.fidelity(
            num_qubits, total_shots=1e4, phi_values=phi_values, noise_model=my_noise_model
        )
        # Each Hadamard including the first one adds a multiplier of (1-p)
        # to the parity oscillation measurements
        expected_fidelity = 0.5 * (1 + (1 - p)**(num_qubits + 1))
        assert abs(par_osc_fid - expected_fidelity) < 1e-2


if __name__ == "__main__":
    unittest.main()
