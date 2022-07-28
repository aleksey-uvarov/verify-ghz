# We take a density matrix of the GHZ state mixed with the maximally mixed state and try to see
# how the different measurement techniques will behave
from main import *
import telescope
import parityoscillations
import multiplequantum
from functools import reduce


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1])


def main():
    alpha = 0.1
    n_qubits = 3
    rho = prepare_noisy_ghz(alpha, n_qubits)
    rho_clean = prepare_noisy_ghz(0, n_qubits)
    print("Exact fidelity ", np.trace(rho @ rho_clean))
    f_tele_low, f_tele_up, stderr_tele_low, stderr_tele_up = get_telescope_fid(rho, 100, 100)
    print("Lower bound ", f_tele_low," +- ", stderr_tele_low)
    print("Upper bound ", f_tele_up," +- ", stderr_tele_up)


def prepare_noisy_ghz(alpha, n_qubits):
    rho_clean = np.zeros((2**n_qubits, 2**n_qubits), dtype='complex')
    rho_clean[0, 0] = 0.5
    rho_clean[0, -1] = 0.5
    rho_clean[-1, 0] = 0.5
    rho_clean[-1, -1] = 0.5
    return (1 - alpha) * rho_clean + alpha * np.eye(2**n_qubits) / 2**n_qubits


def get_all_methods_data(rho: np.array, qty_measurements: int):
    f_telescope, err_telescope = get_telescope_fid(rho, qty_measurements)
    # f_parity, err_parity = get_parity_fid(rho, qty_measurements)
    # f_mqc, err_mqc = get_mqc_fid(rho, qty_measurements)


def get_telescope_fid(rho, measures_x: int, measures_z: int):
    n_qubits = int(np.log2(np.shape(rho)[0]) + 1e-4)
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
    print(probs_zz)

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


def get_parity_fid(rho, qty_measurements):
    pass


def get_mqc_fid(rho, qty_measurements):
    pass


if __name__=="__main__":
    main()