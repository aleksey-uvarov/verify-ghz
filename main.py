import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import erfinv

from qiskit import Aer, execute
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer import AerSimulator


def true_fidelity(circ, noise_model):
    backend = AerSimulator(noise_model=noise_model)
    job = backend.run(circ)
    result = job.result()
    rho = result.results[0].data.density_matrix
    rho_matrix = np.array(rho).reshape((2 ** num_qubits, 2 ** num_qubits))

    backend_clean = AerSimulator(noise_model=None)
    job_clean = backend_clean.run(circ)
    result_clean = job_clean.result()
    state_clean = result_clean.get_statevector()
    return (state_clean.conj() @ rho_matrix @ state_clean).real


def telescope_fidelity(n_qubits, total_shots, ratio=0.5, noise_model=None):
    simulator = Aer.get_backend('qasm_simulator')
    circ_ghz = get_ghz_circuit(n_qubits)

    circ_ghz_measure_z = append_measurements_to_circ(circ_ghz, 'z')
    circ_ghz_measure_x = append_measurements_to_circ(circ_ghz, 'x')

    result_z_noisy = execute(circ_ghz_measure_z, simulator,
                             noise_model=noise_model,
                             shots=total_shots * ratio).result()
    counts_z_noisy = result_z_noisy.get_counts(circ_ghz_measure_z)

    result_x_noisy = execute(circ_ghz_measure_x, simulator,
                             noise_model=noise_model,
                             shots=total_shots * (1 - ratio)).result()
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

    parity_vals, parity_errors = parity_oscillations_data(n_qubits,
                                                          shots_parity, phi_values,
                                                          noise_model)

    fit_amp, fit_phase, fit_amp_error = parity_oscillations_fit(parity_vals,
                                                                parity_errors, phi_values)

    simulator = Aer.get_backend('qasm_simulator')
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    circ = QuantumCircuit(q, c)
    circ.h(q[0])
    for i in range(n_qubits - 1):
        circ.cx(q[i], q[i+1])
    circ.measure(q, c)
    result = execute(circ, simulator,
                     noise_model=noise_model,
                     shots=shots_z).result()
    counts = result.get_counts(circ)

    # following the naming in Omran et al. (2019)
    alphas = (counts['0' * n_qubits] + counts['1' * n_qubits]) / shots_z

    alphas_variance = 1 - alphas**2

    fidelity = 0.5 * (alphas + abs(fit_amp))
    total_error = fit_amp_error + (alphas_variance / shots_z)**0.5

    print('beta', fit_amp, 'alpha', alphas)
    return fidelity, total_error


def parity_oscillations_fit(parity_vals, parity_errors, phi_values):

    def f(x, amp, phase):
        return amp * np.sin(num_qubits * x + phase)

    def residual(p):
        deltas = f(phi_values, p[0], p[1]) - parity_vals
        return deltas / (parity_errors + 1e-3)

    p0 = [1., 0.]
    fit = optimize.least_squares(residual, p0)

    def total_residual_squared(x):
        return np.sum(residual([x, fit.x[1]])**2)

    def confidence_intersection(x):
        return total_residual_squared(x) - total_residual_squared(fit.x[0]) - 1

    sol_1 = optimize.diagbroyden(confidence_intersection, fit.x[0] - 0.1)
    sol_2 = optimize.diagbroyden(confidence_intersection, fit.x[0] + 0.1)
    error = max(abs(sol_1 - fit.x[0]), abs(sol_2 - fit.x[0]))
    # might silently give an error if both converge to the same place!

    return fit.x[0], fit.x[1], error


def parity_oscillations_data(n_qubits, total_shots, phi_values,
                             noise_model=None):
    simulator = Aer.get_backend('qasm_simulator')
    q, c = QuantumRegister(n_qubits), ClassicalRegister(n_qubits)
    circ_ghz = QuantumCircuit(q, c)
    circ_ghz.h(q[0])
    for i in range(n_qubits - 1):
        circ_ghz.cx(q[i], q[i+1])

    #     phi_values = np.linspace(0, 2 * np.pi / n_qubits, num=phi_steps)
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

        result = execute(circ, simulator,
                         noise_model=noise_model,
                         shots=shots_per_point).result()
        counts = result.get_counts(circ)
        parity_counts = np.zeros(2)
        for k, v in counts.items():
            parity = k.count('1')
            parity_counts[parity % 2] += v
        #         E_x += (-1)**(parity + 1) * v
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


if __name__ == "__main__":

    my_noise_model = NoiseModel()
    # Add depolarizing error to all single qubit u1, u2, u3 gates
    p1 = 0.01
    p2 = 0.04
    error = depolarizing_error(p1, 1)
    error_cx = depolarizing_error(p2, 2)
    my_noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3', 'h'])
    my_noise_model.add_all_qubit_quantum_error(error_cx, ['cx'])

    phi_steps = 20
    num_qubits = 5
    phi_values = np.linspace(0, 4 * 2 * np.pi / num_qubits, num=phi_steps)

    n_tests = 5

    def f(x, amp, phase):
        return amp * np.sin(num_qubits * x + phase)

    shot_budgets = np.linspace(20 * (phi_steps + 1), 100 * (phi_steps + 1), num=n_tests)
    telescope_data = np.zeros((n_tests, 4))
    parity_data = np.zeros((n_tests, 2))
    for i, shots in enumerate(shot_budgets):
        f_tele = telescope_fidelity(num_qubits, shots, noise_model=my_noise_model)
        parity_vals, parity_errors = parity_oscillations_data(num_qubits,
                                                              shots, phi_values,
                                                              my_noise_model)
        fit_amp, fit_phase, error = parity_oscillations_fit(parity_vals, parity_errors,
                                                             phi_values)
        f_parity = [fit_amp, error]
        # f_parity = parity_oscillations_fidelity(n_qubits, shots, phi_values, noise_model)
        telescope_data[i, :] = f_tele
        parity_data[i, :] = f_parity

    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'figure.figsize': [9, 6]})


    plt.errorbar(shot_budgets, telescope_data[:, 0], yerr=2*telescope_data[:, 2], label='lower bound')
    plt.errorbar(shot_budgets, telescope_data[:, 1], yerr=2*telescope_data[:, 3], label='upper bound')
    plt.errorbar(shot_budgets, parity_data[:, 0], yerr=2*parity_data[:, 1], label='parity')
    # plt.plot([min(shot_budgets), max(shot_budgets)], [fidelity_true, fidelity_true], '--', color='gray',
    #          label='true fidelity')
    plt.xlabel('Total shots')
    plt.ylabel('Fidelity')
    plt.grid()

    plt.title('$n = {0:}, \phi \in [0, {1:2.1f} \pi]$, {2:} values of $\phi$; \n \
                noise model: p1={3:2.3f}, p2={4:2.3f}'.format(
                num_qubits, max(phi_values) / np.pi, phi_steps, p1, p2))
    plt.legend()
    plt.show()