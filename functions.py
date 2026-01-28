import numpy as np
import matplotlib.pyplot as plt
import pyscf
import math

def calculate_shannon_entropy(ci_coefficients):
    """
    Calculate the Shannon entropy from the CI coefficients.

    Args:
        ci_coefficients (ndarray, flatten): CI coefficients from an FCI calculation.

    Returns:
        float: Shannon entropy of the CI wavefunction.
    """

    # no square
    ci_ab = np.abs(ci_coefficients)

    # Normalization
    # ci_ab /= np.sum(ci_ab)

    # Compute the Shannon entropy
    # Only consider non-zero probabilities to avoid log(0) issues
    non_zero_ci = ci_ab[ci_ab > 0]
    entropy = -np.sum(non_zero_ci * np.log(non_zero_ci))

    return entropy


from warnings import warn


def shannon(spec):
    '''
    Shannon entropy of a probability distribution

    Args:
        spec (ndarray): probability distribution

    Returns:
        S (float): Shannon entropy of spec
    '''
    spec = np.asarray(spec)
    if np.any(spec < 0):
        if np.any(np.abs(spec[spec < 0]) > 1e-6):
            warn("Warning: spec has negative entries!")
    elif np.any(spec > 1):
        print(spec)
        raise ValueError("spec has entries larger than 1")
    spec = spec[spec > 0]
    return -np.sum(spec * np.log(spec))


def get_cost_fqi(gamma, Gamma, inactive_indices):
    '''
    Sum of all inactive orbital entropy

    Args:
        gamma (ndarray): current 1RDM
        Gamma (ndarray): current 2RDM
        inactive_indices (list): indices of inactive orbitals

    Returns:
        cost_fun (float): S(rho_i) for all i in inactive_indices

    '''

    inds = np.asarray(inactive_indices)
    nu = gamma[2 * inds, 2 * inds]
    nd = gamma[2 * inds + 1, 2 * inds + 1]
    nn = Gamma[inds, inds, inds, inds]
    spec = np.array([1 - nu - nd + nn, nu - nn, nd - nn, nn])
    cost_fun = shannon(spec)

    return np.sum(cost_fun)


def prep_rdm12(dm1, dm2):
    '''
    Prepare the 1- and 2-RDM (splitting 1-RDM into spin parts and fix prefactor of 2-RDM)
    This only works for singlet states.
    For other spin states, one should run spin unrestricted DMRG and get the
    spin 1- and 2-RDMs.

    Args:
        dm1 (ndarray): spatial-orbital 1RDM from pyscf
        dm2 (ndarray): spatial-orbital 2RDM from pyscf

    Returns:
        rdm1(ndarray): prepared 1RDM in spin-orbital indices
        rdm2(ndarray): prepared relevant part of the 2RDM in orbital indices and spin (up,down,down,up)
    '''
    no = len(dm1)
    rdm1 = np.zeros((2 * no, 2 * no))
    rdm1[::2, ::2] = dm1 / 2
    rdm1[1::2, 1::2] = dm1 / 2
    rdm2 = dm2.transpose((0, 2, 3, 1)).copy()
    rdm2 = (2 * rdm2 + rdm2.transpose((0, 1, 3, 2))) / 6.

    return rdm1, rdm2

def make_no(cisolver, ci, myhf, mo_coeff=None):
    if mo_coeff is None:
        mo_coeff = myhf.mo_coeff
    rdm1_fci = cisolver.make_rdm1(ci, myhf.mo_coeff.shape[0], myhf.mol.nelectron)
    occ_fci, natural_orbs_fci = np.linalg.eigh(rdm1_fci)
    idx_fci = occ_fci.argsort()[::-1]
    natural_orbs_fci = natural_orbs_fci[:, idx_fci]
    no_coeff_fci = np.dot(mo_coeff, natural_orbs_fci)
    return no_coeff_fci

def plot_data(bond_lengths, data, names_of_orbitals, y_label, title):
    """
    Plot the data for different orbitals.

    Parameters:
    bond_lengths (list or ndarray): The x-axis values (e.g., bond lengths).
    data (ndarray): The y-axis values for different orbitals (shape: (n_orbitals, n_points)).
    names_of_orbitals (list): The labels for different orbitals.
    y_label (str): The label for the y-axis.
    title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 7))