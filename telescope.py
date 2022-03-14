from qiskit.providers.aer import AerSimulator
import numpy as np
import main
from scipy.special import erfinv

import shared


def fidelity(n_qubits, total_shots, ratio=0.5, noise_model=None):
    backend = AerSimulator(noise_model=noise_model)
    circ_ghz = shared.get_ghz_circuit(n_qubits)

    circ_ghz_measure_z = shared.append_measurements_to_circ(circ_ghz, 'z')
    circ_ghz_measure_x = shared.append_measurements_to_circ(circ_ghz, 'x')

    result_z_noisy = backend.run(circ_ghz_measure_z, shots=total_shots * ratio).result()
    counts_z_noisy = result_z_noisy.get_counts(circ_ghz_measure_z)

    result_x_noisy = backend.run(circ_ghz_measure_x, shots=total_shots * (1-ratio)).result()
    counts_x_noisy = result_x_noisy.get_counts(circ_ghz_measure_x)

    energy_z, error_z = energy_and_error_z(counts_z_noisy)
    energy_x, error_x = energy_and_error_x(counts_x_noisy)
    energy = energy_z + energy_x
    energy_error = error_z + error_x

    fidelity_lower_bound = 1 - energy / 2
    fidelity_lower_error = energy_error / 2
    fidelity_upper_bound = 1 - energy / (2 * n_qubits)
    fidelity_upper_error = energy_error / (2 * n_qubits)
    return (fidelity_lower_bound, fidelity_upper_bound,
            fidelity_lower_error, fidelity_upper_error)


def ghz_tailored_fidelity(n_qubits, total_shots, ratio=0.5, noise_model=None):
    pop, pop_stderr = shared.fidelity_population(n_qubits, total_shots * ratio, noise_model)
    backend = AerSimulator(noise_model=noise_model)
    circ_ghz = shared.get_ghz_circuit(n_qubits)
    circ_ghz_measure_x = shared.append_measurements_to_circ(circ_ghz, 'x')
    result_x_noisy = backend.run(circ_ghz_measure_x, shots=total_shots * (1-ratio)).result()
    counts_x_noisy = result_x_noisy.get_counts(circ_ghz_measure_x)
    expected_x, error_x = expected_xxx_and_error(counts_x_noisy)

    print('pop ', pop)
    print('xxx ', expected_x)

    lower_bound = pop + expected_x - 1
    lower_error = pop_stderr + error_x
    upper_bound = (pop + expected_x + 1) / 3
    upper_error = (pop_stderr + error_x) / 3
    return (lower_bound, upper_bound,
            lower_error, upper_error)


def expected_xxx_and_error(counts_x):
    """Returns the expected value of XXXXX and std error based on the measurements in the X basis"""
    shots_x = sum(counts_x.values())
    parity_counts = np.zeros(2)
    for k, v in counts_x.items():
        parity = k.count('1')
        parity_counts[parity % 2] += v
    parity_probabilities = parity_counts / shots_x
    energy = parity_probabilities[0] - parity_probabilities[1]
    energy_error = ((1 - energy ** 2) / shots_x) ** 0.5
    return energy, energy_error


def energy_and_error_z(counts_z):
    shots_z = sum(counts_z.values())
    n = 0  #
    for k in counts_z:
        n = len(k)
        break
    flip_counts = np.zeros(n+1)
    for k, v in counts_z.items():
        n_flips = 0
        for j in range(len(k) - 1):
            if k[j] != k[j + 1]:
                n_flips += 1
        flip_counts[n_flips] += v

    flip_probabilities = flip_counts / shots_z
    avg_flip_count = np.sum(np.arange(n+1) * flip_probabilities)
    flip_variance = (np.sum(np.arange(n+1) * flip_probabilities * np.arange(n+1))
                     - avg_flip_count**2)
    energy_z = 2 * avg_flip_count
    energy_z_variance = 4 * flip_variance
    std_error = (energy_z_variance / shots_z)**0.5
    return energy_z, std_error


def energy_z_with_confidence(counts_z, confidence_level=0.95):
    shots_z = sum(counts_z.values())
    n = 0  #
    for k in counts_z:
        n = len(k)
        break
    flip_counts = np.zeros(n+1)
    for k, v in counts_z.items():
        n_flips = 0
        for j in range(len(k) - 1):
            if k[j] != k[j + 1]:
                n_flips += 1
        flip_counts[n_flips] += v

    flip_probabilities = flip_counts / shots_z
    avg_flip_count = np.sum(np.arange(n+1) * flip_probabilities)
    flip_variance = (np.sum(np.arange(n+1) * flip_probabilities * np.arange(n+1))
                     - avg_flip_count**2)
    energy_z = 2 * avg_flip_count
    energy_z_variance = 4 * flip_variance
    std_error = (energy_z_variance / shots_z)**0.5
    confidence_interval_size = std_error * erfinv(confidence_level) * 2 ** 0.5
    return energy_z, [confidence_interval_size, confidence_interval_size]


def energy_and_error_x(counts_x):
    """Calculates the expected value and std error of parity times 2,
    i.e. -X...X + 1."""
    shots_x = sum(counts_x.values())
    parity_counts = np.zeros(2)
    for k, v in counts_x.items():
        parity = k.count('1')
        parity_counts[parity % 2] += v
    parity_probabilities = parity_counts / shots_x
    energy_x = 2 * parity_probabilities[1]
    energy_x_variance = 4 * (parity_probabilities[1] - parity_probabilities[1]**2)
    std_error = (energy_x_variance / shots_x)**0.5
    return energy_x, std_error


def energy_x_with_confidence(counts_x, confidence_level=0.95):
    shots_x = sum(counts_x.values())
    parity_counts = np.zeros(2)
    for k, v in counts_x.items():
        parity = k.count('1')
        parity_counts[parity % 2] += v
    parity_probabilities = parity_counts / shots_x
    energy_x = 2 * parity_probabilities[1]
    energy_x_variance = 4 * (parity_probabilities[1] - parity_probabilities[1]**2)
    std_error = (energy_x_variance / shots_x)**0.5
    return energy_x, std_error
