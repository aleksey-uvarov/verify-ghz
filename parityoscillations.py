from scipy import optimize
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import AerSimulator
import numpy as np

import main
import shared


def fidelity(n_qubits, total_shots, phi_values, noise_model=None):
    shots_z = total_shots // (len(phi_values) + 1)
    shots_parity = total_shots - shots_z

    coh, coh_error = coherence_with_bootstrap(n_qubits, shots_parity,
                                              phi_values, noise_model=noise_model)
    alphas, alphas_variance = shared.fidelity_population(n_qubits, shots_z, noise_model)

    fid = 0.5 * (alphas + coh)
    total_error = coh_error + (alphas_variance / shots_z)**0.5

    return fid, total_error


def coherence(n_qubits, shots_parity, phi_values,
              noise_model=None):
    parity_vals, parity_errors = parity_oscillations_data(n_qubits,
                                                          shots_parity, phi_values,
                                                          noise_model)
    fit_function = shared.make_parametrized_cosine(n_qubits)
    popt, pcov = optimize.curve_fit(fit_function, phi_values, parity_vals, sigma=parity_errors)
    return abs(popt[0]), pcov[0, 0]


def coherence_with_bootstrap(n_qubits, shots_parity, phi_values,
                             n_bootstraps=100, noise_model=None):
    parity_vals, _ = parity_oscillations_data(n_qubits,
                                              shots_parity, phi_values,
                                              noise_model)
    fit_function = shared.make_parametrized_cosine(n_qubits)
    popt, _ = optimize.curve_fit(fit_function, phi_values, parity_vals)
    coh_data_estimate = abs(popt[0])
    coh_bootstrap_estimates = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        freqs = shared.sample_binomials((parity_vals + 1) / 2, shots_parity)
        popt, _ = optimize.curve_fit(fit_function, phi_values, 2 * freqs - 1)
        coh_bootstrap_estimates[i] = popt[0]
    bootstrap_variance = np.var(coh_bootstrap_estimates)
    return coh_data_estimate, bootstrap_variance**0.5
    # Efron and Tibshirani give a 'rule of thumb' that 100 bootstrap replications
    # give satisfactory results. Also, here the distributions are thin-tailed or even finite


def parity_oscillations_data(n_qubits, total_shots, phi_values,
                             noise_model=None):
    """Measure the parity in points specified by phi_values,
    using a budget of total_shots for all of them."""
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
        parity_probabilities = parity_counts / shots_per_point
        total_parity = (-1) * parity_probabilities[0] + 1 * parity_probabilities[1]
        parity_variance = 1 - total_parity**2
        parity_vals[i] = total_parity
        parity_errors[i] = (parity_variance / shots_per_point)**0.5
    return parity_vals, parity_errors




