from qiskit.providers.aer import AerSimulator
import numpy as np

import main
import shared


def fidelity(n_qubits, noise_model, shots_population, shots_coherence):
    """Models the MQC fidelity estimation protocol
    from Phys Rev A 101, 032343.
    """
    population, sigma_population = main.fidelity_population(n_qubits, shots_population, noise_model)
    c, sigma_c = coherence(n_qubits, noise_model, shots_coherence)
    return 0.5 * (population + c), 0.5 * (sigma_population + sigma_c)


def coherence(n_qubits, noise_model, shots_coherence):
    # inverse fft comes from different conventions
    # in numpy and in the PRA paper
    coh_signal, coh_sigmas = coherence_signal(n_qubits, noise_model, shots_coherence)
    i_q = np.fft.ifft(coh_signal)
    c = 2 * i_q[n_qubits].real**0.5
    sigma_c = 2 * np.mean(coh_sigmas)**0.5
    return c, sigma_c


def coherence_with_bootstrap(n_qubits, noise_model, shots, n_bootstraps=100):
    coh_signal, _ = coherence_signal(n_qubits, noise_model, shots)
    i_q = np.fft.ifft(coh_signal)
    c = 2 * i_q[n_qubits].real**0.5
    bootstrap_estimates = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        bootstrap_signal = shared.sample_binomials(coh_signal, shots)
        i_q = np.fft.ifft(bootstrap_signal)
        bootstrap_estimates[i] = 2 * i_q[n_qubits].real**0.5
    bootstrap_variance = np.var(bootstrap_estimates)
    return c, (bootstrap_variance / n_bootstraps)**0.5


def coherence_signal(n_qubits, noise_model, shots_coherence):
    """Calculate the probabilities to measure '0' on the first qubit
    in the MQC experiment for relevant values of phi"""
    backend = AerSimulator(noise_model=noise_model)
    phi_values = np.linspace(0, 2 * np.pi, num=(2 * n_qubits + 2), endpoint=False)
    shots_per_experiment = shots_coherence / (2*n_qubits+2)
    signal = np.zeros_like(phi_values)
    for i, phi in enumerate(phi_values):
        circ = mqc_ghz_circuit(n_qubits, phi)
        circ.measure(circ.qregs[0][0], circ.cregs[0][0])
        result = backend.run(circ, shots=shots_per_experiment).result()
        counts = result.get_counts()
        for k, v in counts.items():
            if k[-1] == "0":
                signal[i] += v / shots_per_experiment
    sigmas = ((1 - signal**2) / shots_per_experiment)**0.5
    return signal, sigmas


def mqc_ghz_circuit(n_qubits, phi, hahn_flip=False):
    circ = shared.get_ghz_circuit(n_qubits)
    q = circ.qregs[0]
    for i in range(n_qubits):
        if hahn_flip:
            circ.x(q[i])
        circ.rz(phi, q[i])
    # I'm not implementing parallel preparation of GHZ yet
    for i in range(n_qubits - 1, 0, -1):
        circ.cx(q[i - 1], q[i])
    circ.h(q[0])
    return circ
