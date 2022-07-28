# We take a density matrix of the GHZ state mixed with the maximally mixed state and try to see
# how the different measurement techniques will behave
from main import *
# import telescope
from scipy.optimize import curve_fit
# import multiplequantum
from functools import reduce
import matplotlib.pyplot as plt

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])


def main():
    n_qubits = 5
    measures_x = 500
    measures_z = 500

    measures_coherence = measures_x
    measures_pop = measures_z
    phi_max = 2 * np.pi
    pts_coherence = 10

    # alphas = np.linspace(0, 1, num=21)
    alphas = np.array([0])
    rho_clean = prepare_noisy_ghz(0, n_qubits)
    exact_fidelities = np.zeros_like(alphas)
    telescope_data = np.zeros((alphas.shape[0], 4))

    po_data = np.zeros((alphas.shape[0], 2))

    for i, alpha in enumerate(alphas):
        rho = prepare_noisy_ghz(alpha, n_qubits)
        f_tele_low, f_tele_up, stderr_tele_low, stderr_tele_up = get_telescope_fid(rho, measures_x, measures_z)
        telescope_data[i, :] = [f_tele_low, f_tele_up, stderr_tele_low, stderr_tele_up]
        exact_fidelities[i] = np.trace(rho @ rho_clean).real
        # print("Exact fidelity ", np.trace(rho @ rho_clean))
        po_data[i, :] = get_parity_fid(rho, measures_pop, measures_coherence, pts_coherence, phi_max)
        # print("Lower bound ", f_tele_low," +- ", stderr_tele_low)
        # print("Upper bound ", f_tele_up," +- ", stderr_tele_up)

    plt.plot(alphas, exact_fidelities)
    plt.errorbar(alphas, telescope_data[:, 0], yerr=telescope_data[:, 2])
    plt.errorbar(alphas, telescope_data[:, 1], yerr=telescope_data[:, 3])
    plt.errorbar(alphas, po_data[:, 0], yerr=po_data[:, 1])
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Fidelity')
    plt.ylim([-0.1, 1.1])
    plt.show()


def prepare_noisy_ghz(alpha, n_qubits):
    rho_clean = np.zeros((2**n_qubits, 2**n_qubits), dtype='complex')
    rho_clean[0, 0] = 0.5
    rho_clean[0, -1] = 0.5
    rho_clean[-1, 0] = 0.5
    rho_clean[-1, -1] = 0.5
    return (1 - alpha) * rho_clean + alpha * np.eye(2**n_qubits) / 2**n_qubits


# def get_all_methods_data(rho: np.array, qty_measurements: int):
#     f_telescope, err_telescope = get_telescope_fid(rho, qty_measurements)
#     # f_parity, err_parity = get_parity_fid(rho, qty_measurements)
#     # f_mqc, err_mqc = get_mqc_fid(rho, qty_measurements)


def get_telescope_fid(rho, measures_x: int, measures_z: int):
    n_qubits = int(round(np.log2(np.shape(rho)[0])))
    x_list = [X] * n_qubits
    zz_list = [np.eye(2)] * (n_qubits - 2) + [Z, Z]
    h = np.eye(2**n_qubits, dtype='complex') * n_qubits
    h_x = reduce(np.kron, x_list)
    h_zz = np.zeros_like(rho, dtype='complex')
    for i in range(n_qubits - 1):
        h_zz += reduce(np.kron, zz_list[i:] + zz_list[:i])
    h += - h_x - h_zz

    prob_x = np.trace(rho @ (np.eye(2**n_qubits) + h_x) * 0.5).real

    probs_zz = np.zeros(n_qubits)
    eigvals_zz = -(n_qubits - 1) + 2 * np.arange(n_qubits)
    for i in range(n_qubits):
        indices = np.where(np.diag(h_zz) == eigvals_zz[i])
        probs_zz[i] = np.sum(np.diag(rho)[indices]).real

    rng = np.random.default_rng()
    counts_x = rng.binomial(measures_x, prob_x)
    prob_x_measured = (counts_x / measures_x)
    e_x_measured = 2 * prob_x_measured - 1
    var_x_measured = 4 * prob_x_measured * (1 - prob_x_measured)

    samples_z = rng.choice(eigvals_zz, p=probs_zz, size=(measures_z,))
    e_z_measured = np.mean(samples_z)
    var_z_measured = np.var(samples_z)
    stderr = (var_x_measured / measures_x + var_z_measured / measures_z)**0.5
    e_measured = -e_x_measured - e_z_measured + n_qubits
    fidelity_lower_bound = 1 - e_measured / 2
    fidelity_lower_error = stderr / 2
    fidelity_upper_bound = 1 - e_measured / (2 * n_qubits)
    fidelity_upper_error = stderr / (2 * n_qubits)
    return (fidelity_lower_bound, fidelity_upper_bound,
            fidelity_lower_error, fidelity_upper_error)


def get_parity_fid(rho, measures_pop, measures_coherence, pts_coherence, max_phi):
    n_qubits = int(round(np.log2(np.shape(rho)[0])))
    rng = np.random.default_rng()

    pop_expected = (rho[0, 0] + rho[1, 1]).real
    pop_counts = rng.binomial(measures_pop, pop_expected)
    pop_measured = pop_counts / measures_pop
    pop_var_measured = pop_measured - pop_measured**2

    z_string = reduce(np.kron, [Z] * n_qubits)

    phis = np.linspace(0, max_phi, num=pts_coherence)
    parity_measures = np.zeros_like(phis)
    parity_errors = np.zeros_like(phis)
    for i, phi in enumerate(phis):
        u = 1/2**0.5 * (np.eye(2) - 1j * (np.cos(phi) * X + np.sin(phi) * Y))
        total_unitary = reduce(np.kron, [u] * n_qubits)
        parity_op = total_unitary.T.conj() @ z_string @ total_unitary
        projector = 0.5 * (np.eye(2**n_qubits) + parity_op)
        prob = np.trace(projector @ rho).real  # I think there is a problem with the calculation of probs
        shots = measures_coherence // pts_coherence
        parity_counts = rng.binomial(shots, prob)
        prob_measured = parity_counts / shots
        parity_measured = 2 * prob_measured - 1
        parity_variance = 1 - parity_measured**2 #### Check this twice
        parity_measures[i] = parity_measured
        parity_errors[i] = (parity_variance / shots)**0.5
    print(parity_measures)
    plt.errorbar(phis, parity_measures, yerr=parity_errors)
    plt.show()

    fit_function = shared.make_parametrized_cosine(n_qubits)
    popt, pcov = curve_fit(fit_function, phis, parity_measures, sigma=parity_errors)
    return (abs(popt[0]) + pop_measured) / 2, (pcov[0, 0]**2 + pop_var_measured / measures_pop)**0.5




def get_mqc_fid(rho, qty_measurements):
    pass


if __name__ == "__main__":
    main()
