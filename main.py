import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from scipy import optimize

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer import AerSimulator, StatevectorSimulator

import multiplequantum
import parityoscillations
import telescope


def parity_osc_coherence_errors(qubit_numbers: np.array, phi_point_numbers: np.array,
                                total_shots_parity=1e5, noise_model=None):
    error_values = np.zeros((len(qubit_numbers), len(phi_point_numbers)))
    for i, n in enumerate(qubit_numbers):
        for j, m in enumerate(phi_point_numbers):
            phi_values = np.linspace(0, 2 * np.pi, num=m)
            _, error_values[i, j] = parityoscillations.coherence(n, total_shots_parity,
                                                                 phi_values, noise_model)
    return error_values


def population_error_experiment(n_qubits: int, shot_numbers: Iterable,
                                noise_model: NoiseModel = None):
    sigmas = []
    for i, n_shots in enumerate(shot_numbers):
        _, sigma = fidelity_population(n_qubits, n_shots, noise_model)
        sigmas.append(sigma)
    return sigmas


def compare_fidelity_estimates():
    n_qubits = 4
    shots_coherence = 1e5
    shots_pop = 1e5

    mqc_fid, mqc_err = multiplequantum.fidelity(n_qubits, my_noise_model, shots_pop, shots_coherence)
    f_true = true_fidelity(get_ghz_circuit(n_qubits), my_noise_model)
    po_fid, po_err = parityoscillations.fidelity(n_qubits, shots_coherence + shots_pop,
                                                 np.linspace(0, 2 * np.pi, num=2 * n_qubits + 2), my_noise_model)
    telescope_data = telescope.fidelity(n_qubits,
                                        shots_coherence + shots_pop,
                                        noise_model=my_noise_model)
    print("True ", f_true)
    print("MQC ", mqc_fid, mqc_err)
    print("PO ", po_fid, po_err)
    print("Telescope lower ", telescope_data[0], telescope_data[2])
    print("Telescope upper ", telescope_data[1], telescope_data[3])


def true_fidelity(circ, noise_model):
    backend = AerSimulator(noise_model=noise_model)

    backend_clean = StatevectorSimulator()
    job_clean = backend_clean.run(circ)
    result_clean = job_clean.result()
    state_clean = np.array(result_clean.get_statevector())

    circ.save_density_matrix()
    job = backend.run(circ)
    result = job.result()
    rho = result.results[0].data.density_matrix

    return (state_clean.conj() @ rho.data @ state_clean).real


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


def fidelity_population(n_qubits, n_shots, noise_model):
    backend = AerSimulator(noise_model=noise_model)
    circ = append_measurements_to_circ(get_ghz_circuit(n_qubits), 'z')
    result = backend.run(circ, shots=n_shots, seed=100).result()
    counts = result.get_counts(circ)

    # following the naming in Omran et al. (2019)
    alphas = (counts['0' * n_qubits] + counts['1' * n_qubits]) / n_shots
    alphas_variance = 1 - alphas**2
    return alphas, (alphas_variance / n_shots)**0.5  # again maybe wilson?


def make_parametrized_cosine(n_qubits):

    def parametrized_cosine(x: np.array, amp: float, phase: float):
        return amp * np.cos(n_qubits * x - phase)

    return parametrized_cosine


def depolarizing_noise_model(p_single, p_cx):
    noise_model = NoiseModel()
    # Add depolarizing error to all single qubit u1, u2, u3 gates
    error = depolarizing_error(p_single, 1)
    error_cx = depolarizing_error(p_cx, 2)
    noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h'])
    noise_model.add_all_qubit_quantum_error(error_cx, ['cx'])
    return noise_model


if __name__ == "__main__":
    p1 = 1e-2
    p2 = 1e-1
    shots_parity = 1e5
    my_noise_model = depolarizing_noise_model(p1, p2)

    shots_numbers = np.linspace(1000, 1e4, num=10)
    shots_dense = np.linspace(min(shots_numbers), max(shots_numbers), num=100)
    num_qubits = 12
    population_sigmas = population_error_experiment(num_qubits, shots_numbers,
                                                    noise_model=my_noise_model)
    print(population_sigmas)
    popt, pcov = optimize.curve_fit(lambda x, a: a * x**(-0.5), shots_numbers, population_sigmas)
    plt.loglog(shots_numbers, population_sigmas, 'o')
    plt.loglog(shots_dense, popt[0] * shots_dense**(-0.5))
    print(popt)
    plt.grid()
    plt.show()
