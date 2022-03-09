import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
import time

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator, StatevectorSimulator

import multiplequantum
import parityoscillations
import shared
import telescope
from shared import get_ghz_circuit, append_measurements_to_circ, depolarizing_noise_model
from shared import only_h_noise_model


def error_as_nq_experiment(qubit_numbers, n_shots=2**15, noise_model=None):
    po_coh_errors = np.zeros(len(qubit_numbers))
    mqc_coh_errors = np.zeros(len(qubit_numbers))
    pop_errors = np.zeros(len(qubit_numbers))
    lower_errors = np.zeros(len(qubit_numbers))
    upper_errors = np.zeros(len(qubit_numbers))

    t0 = time.time()
    for i, n in enumerate(qubit_numbers):
        print(int(time.time() - t0))
        print(n, " qubits...")
        phi_values = np.linspace(0, 2 * np.pi, num=2 * n + 2, endpoint=False)
        _, po_coh_errors[i] = parityoscillations.coherence_with_bootstrap(n, n_shots,
                                                                          phi_values, noise_model=noise_model)
        _, mqc_coh_errors[i] = multiplequantum.coherence_with_bootstrap(n_qubits=n, noise_model=noise_model,
                                                                        shots=n_shots)
        _, pop_errors[i] = fidelity_population(n, n_shots, noise_model)
        _, __, lower_errors[i], upper_errors[i] = telescope.fidelity(n, n_shots, noise_model=noise_model)

    markersize = 8
    t = int(time.time())

    plt.figure()
    plt.plot(qubit_numbers, pop_errors, 's--', label='pop', ms=markersize, color='tab:purple')
    plt.plot(qubit_numbers, po_coh_errors, 'o--', label='po', ms=markersize, color='tab:orange')
    plt.plot(qubit_numbers, lower_errors, '^-', label='low', ms=markersize, color='tab:green')
    plt.plot(qubit_numbers, upper_errors, 'v-', label='high', ms=markersize, color='tab:red')
    plt.plot(qubit_numbers, mqc_coh_errors, 'h--', label='mqc', ms=markersize, color='tab:blue')
    plt.xlabel('Qubits')
    plt.ylabel('Error')
    plt.xticks(qubit_numbers, labels=[str(n) for n in qubit_numbers])
    # plt.legend()
    plt.grid()
    plt.savefig('errors_as_nq_{0:}.eps'.format(t), format='eps', bbox_inches='tight')
    plt.show()


def all_errors_experiment(shot_numbers, n_qubits, noise_model=None):
    """Plot the errors of population, PO coherence, MQ coherence, and
    Hamiltonian bounds as functions of the number of qubits"""
    pop_vals = np.zeros_like(shot_numbers)
    pop_errs = np.zeros_like(shot_numbers)
    po_coh_vals = np.zeros_like(shot_numbers)
    po_coh_errs = np.zeros_like(shot_numbers)
    mq_coh_vals = np.zeros_like(shot_numbers)
    mq_coh_errs = np.zeros_like(shot_numbers)
    lower_vals = np.zeros_like(shot_numbers)
    lower_errs = np.zeros_like(shot_numbers)
    upper_vals = np.zeros_like(shot_numbers)
    upper_errs = np.zeros_like(shot_numbers)

    phi_values = np.linspace(0, 2 * np.pi, num=2 * n_qubits + 2, endpoint=False)
    for i, shots in enumerate(shot_numbers):
        pop_vals[i], pop_errs[i] = fidelity_population(n_qubits, shots, noise_model)
        po_coh_vals[i], po_coh_errs[i] = parityoscillations.coherence_with_bootstrap(n_qubits,
                                                                                     shots,
                                                                                     phi_values,
                                                                                     noise_model=noise_model,
                                                                                     n_bootstraps=100)
        mq_coh_vals[i], mq_coh_errs[i] = multiplequantum.coherence_with_bootstrap(n_qubits, noise_model, shots)
        lower_vals[i], upper_vals[i], lower_errs[i], upper_errs[i] = telescope.fidelity(n_qubits, shots,
                                                                                        ratio=0.5,
                                                                                        noise_model=noise_model)

    f_true = true_fidelity(shared.get_ghz_circuit(n_qubits), noise_model)

    labels = ['$2^{{{0:}}}$'.format(int(np.log2(s))) for s in shot_numbers]
    markersize = 8
    t = int(time.time())

    plt.figure()
    plt.errorbar(shot_numbers, pop_vals, yerr=pop_errs, label='Population',
                 marker='s', ms=markersize, color='tab:purple')
    plt.errorbar(shot_numbers, po_coh_vals, yerr=po_coh_errs, label='Coherence PO',
                 marker='o', ls='--', ms=markersize, color='tab:orange')
    plt.errorbar(shot_numbers, mq_coh_vals, yerr=mq_coh_errs, label='Coherence MQ',
                 marker='h', ls='--', ms=markersize, color='tab:blue')
    plt.errorbar(shot_numbers, lower_vals, yerr=lower_errs, label='H lower',
                 marker='^', ls='-', ms=markersize, color='tab:green')
    plt.errorbar(shot_numbers, upper_vals, yerr=upper_errs, label='H upper',
                 marker='v', ls='-', ms=markersize, color='tab:red')
    plt.xlabel('Shots')
    plt.ylabel('Fidelity')
    ax = plt.axes()
    ax.set_xscale("log")
    plt.xticks(shot_numbers, labels=labels)
    # plt.legend()
    plt.savefig('fidelity_terms_{0:}_qubits_{1:}.eps'.format(n_qubits, t), format='eps', bbox_inches='tight')


# This plot makes no sense unless you fix the shot numbers somehow.
    # Effectively for fidelities we use up a double number of shots.
    # plt.figure()
    # plt.errorbar(shot_numbers, (po_coh_vals + pop_vals)/2,
    #              yerr=(po_coh_errs + pop_errs)/2, label='PO',
    #              marker='o', ls='--', ms=markersize, color='tab:orange')
    # plt.errorbar(shot_numbers, (mq_coh_vals + pop_vals)/2,
    #              yerr=(mq_coh_errs + pop_errs)/2, label='MQC',
    #              marker='h', ls='--', ms=markersize, color='tab:blue')
    # plt.errorbar(shot_numbers, lower_vals,
    #              yerr=lower_errs, label='H lower',
    #              marker='^', ls='-', ms=markersize, color='tab:green')
    # plt.errorbar(shot_numbers, upper_vals,
    #              yerr=upper_errs, label='H upper',
    #              marker='v', ls='-', ms=markersize, color='tab:red')
    # plt.xlabel('Shots')
    # plt.ylabel('Fidelity')
    # ax = plt.axes()
    # ax.set_xscale("log")
    # plt.xticks(shot_numbers, labels=labels)
    # # I draw a line that crosses all the viewport.
    # # To do that, I record current lims, plot, then go back to the lims
    # # (otherwise pyplot would change the viewport to accomodate)
    # xlim = plt.xlim()
    # ylim = plt.ylim()
    # plt.plot([0, 10 * max(shot_numbers)], [f_true, f_true], '-.', color='black')
    # plt.xlim(xlim)
    # plt.ylim(ylim)
    # # plt.legend()
    # plt.savefig('fidelity_{0:}_qubits.eps'.format(n_qubits), format='eps', bbox_inches='tight')

    plt.figure()
    plt.loglog(shot_numbers, pop_errs, 's--', label='pop', ms=markersize, color='tab:purple')
    plt.loglog(shot_numbers, po_coh_errs, 'o--', label='po', ms=markersize, color='tab:orange')
    plt.loglog(shot_numbers, lower_errs, '^-', label='low', ms=markersize, color='tab:green')
    plt.loglog(shot_numbers, upper_errs, 'v-', label='high', ms=markersize, color='tab:red')
    plt.loglog(shot_numbers, mq_coh_errs, 'h--', label='mqc', ms=markersize, color='tab:blue')
    plt.xlabel('Shots')
    plt.ylabel('Error')
    plt.xticks(shot_numbers, labels=labels)
    # plt.legend()
    plt.grid()
    plt.savefig('errors_{0:}_qubits_{1:}.eps'.format(n_qubits, t), format='eps', bbox_inches='tight')
    plt.show()


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


def fidelity_population(n_qubits, n_shots, noise_model):
    backend = AerSimulator(noise_model=noise_model)
    circ = append_measurements_to_circ(get_ghz_circuit(n_qubits), 'z')
    result = backend.run(circ, shots=n_shots, seed=100).result()
    counts = result.get_counts(circ)

    # following the naming in Omran et al. (2019)
    alphas = (counts['0' * n_qubits] + counts['1' * n_qubits]) / n_shots
    alphas_variance = 1 - alphas**2
    return alphas, (alphas_variance / n_shots)**0.5  # again maybe wilson?


if __name__ == "__main__":
    plt.rcParams.update({'figure.figsize': (9, 6), 'font.size': 18})
    p1 = 2e-2
    p2 = 0
    # my_noise_model = depolarizing_noise_model(p1, p2)
    my_noise_model = only_h_noise_model(p1)

    # all_errors_experiment(n_qubits=4,
    #                       noise_model=my_noise_model,
    #                       shot_numbers=np.logspace(11, 16, num=6, base=2))

    my_qubit_numbers = np.arange(4, 16)
    error_as_nq_experiment(my_qubit_numbers, noise_model=my_noise_model)
