import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import erfinv

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer import AerSimulator, StatevectorSimulator


def true_fidelity(circ, noise_model):
    backend = AerSimulator(noise_model=noise_model)

    backend_clean = StatevectorSimulator()
    job_clean = backend_clean.run(circ)
    result_clean = job_clean.result()
    # get_statevector returns something that's not a vector, splendid
    state_clean = np.array(result_clean.get_statevector())

    circ.save_density_matrix()
    job = backend.run(circ)
    result = job.result()
    rho = result.results[0].data.density_matrix
    rho_nq = len(rho.shape) // 2
    rho_matrix = np.array(rho).reshape((2 ** rho_nq, 2 ** rho_nq))

    return (state_clean.conj() @ rho_matrix @ state_clean).real


def telescope_fidelity(n_qubits, total_shots, ratio=0.5, noise_model=None):
    backend = AerSimulator(noise_model=noise_model)
    circ_ghz = get_ghz_circuit(n_qubits)

    circ_ghz_measure_z = append_measurements_to_circ(circ_ghz, 'z')
    circ_ghz_measure_x = append_measurements_to_circ(circ_ghz, 'x')

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


def get_ghz_circuit(n_qubits):
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    circ_ghz = QuantumCircuit(q, c)
    circ_ghz.h(q[0])
    for i in range(n_qubits - 1):
        circ_ghz.cx(q[i], q[i+1])
    return circ_ghz


def append_measurements_to_circ(circ, which):
    if which not in ('z', 'x'):
        raise ValueError("which must be 'z' or 'x'")
    q, c = circ.qregs[0], circ.cregs[0]
    circ_measure = circ.compose(QuantumCircuit(q, c))
    if which == 'x':
        for i in range(len(q)):
            circ_measure.h(q[i])
    circ_measure.measure(q, c)
    return circ_measure


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


def wilson_score_interval(p, n_counts, confidence_level):
    """Not sure how important that is, so
    maybe later"""
    pass


def parity_oscillations_fidelity(n_qubits, total_shots, phi_values,
                                 noise_model=None):

    shots_z = total_shots // (len(phi_values) + 1)
    shots_parity = total_shots - shots_z

    coherence, coherence_error = parity_oscillations_coherence(n_qubits, shots_parity,
                                                               phi_values, noise_model)
    alphas, alphas_variance = fidelity_population(n_qubits, shots_z, noise_model)

    fidelity = 0.5 * (alphas + coherence)
    total_error = coherence_error + (alphas_variance / shots_z)**0.5

    return fidelity, total_error


def parity_oscillations_coherence(n_qubits, shots_parity, phi_values,
    noise_model=None):
    parity_vals, parity_errors = parity_oscillations_data(n_qubits,
                                                          shots_parity, phi_values,
                                                          noise_model)
    fit_function = make_parametrized_cosine(n_qubits)
    popt, pcov = optimize.curve_fit(fit_function, phi_values, parity_vals, sigma=parity_errors)
    return abs(popt[0]), pcov[0, 0]


def fidelity_population(n_qubits, n_shots, noise_model):
    backend = AerSimulator(noise_model=noise_model)
    circ = append_measurements_to_circ(get_ghz_circuit(n_qubits), 'z')
    result = backend.run(circ, shots=n_shots, seed=100).result()
    counts = result.get_counts(circ)

    # following the naming in Omran et al. (2019)
    alphas = (counts['0' * n_qubits] + counts['1' * n_qubits]) / n_shots
    alphas_variance = 1 - alphas**2
    return alphas, (alphas_variance / n_shots)**0.5 # again maybe wilson?


def make_parametrized_cosine(num_qubits):

    def parametrized_cosine(x: np.array, amp: float, phase: float):
        return amp * np.cos(num_qubits * x - phase)

    return parametrized_cosine


def parity_oscillations_data(n_qubits, total_shots, phi_values,
                             noise_model=None):
    backend = AerSimulator(noise_model=noise_model)
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    circ_ghz = QuantumCircuit(q, c)
    circ_ghz.h(q[0])
    for i in range(n_qubits - 1):
        circ_ghz.cx(q[i], q[i+1])

    phi_steps = len(phi_values)
    parity_vals = np.zeros_like(phi_values)
    parity_errors = np.zeros_like(phi_values)
    shots_per_point = total_shots / phi_steps
    for i, phi in enumerate(phi_values):
        circ = QuantumCircuit(q, c)
        for j in range(n_qubits):
            circ.rz(phi, q[j])
            circ.h(q[j])
        circ.measure(q, c)
        circ = circ_ghz.compose(circ)
        result = backend.run(circ, shots=shots_per_point).result()

        counts = result.get_counts(circ)
        parity_counts = np.zeros(2)
        for k, v in counts.items():
            parity = k.count('1')
            parity_counts[parity % 2] += v
        parity_counts = parity_counts / shots_per_point
        total_parity = (-1) * parity_counts[0] + 1 * parity_counts[1]
        parity_variance = 1 - total_parity**2
        parity_vals[i] = total_parity
        parity_errors[i] = (parity_variance / shots_per_point)**0.5
    return parity_vals, parity_errors
    # todo: should return parity_vals, parity_ci
    # the latter to be calculated by Wilson or similar


def measure_parity(n_qubits, n_shots, phi, noise_model=None):
    pass


def parity_osc_coherence_errors(qubit_numbers: np.array, phi_point_numbers: np.array,
                                total_shots_parity=1e5, noise_model=None):
    error_values = np.zeros((len(qubit_numbers), len(phi_point_numbers)))
    for i, n in enumerate(qubit_numbers):
        for j, m in enumerate(phi_point_numbers):
            phi_values = np.linspace(0, 2 * np.pi, num=m)
            _, error_values[i, j] = parity_oscillations_coherence(n, total_shots_parity,
                                                                  phi_values, noise_model)
    return error_values


def depolarizing_noise_model(p1, p2):
    noise_model = NoiseModel()
    # Add depolarizing error to all single qubit u1, u2, u3 gates
    error = depolarizing_error(p1, 1)
    error_cx = depolarizing_error(p2, 2)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h'])
    noise_model.add_all_qubit_quantum_error(error_cx, ['cx'])
    return noise_model


if __name__ == "__main__":
    p1 = 1e-3
    p2 = 1e-2
    shots_parity = 1e5
    qubit_numbers = np.arange(2, 5)
    phi_point_numbers = np.array([20, 30, 40, 50])

    my_noise_model = depolarizing_noise_model(p1, p2)
    error_values = parity_osc_coherence_errors(qubit_numbers, phi_point_numbers,
                                               shots_parity, my_noise_model)

    print(error_values)


