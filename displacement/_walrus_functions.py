import numpy as np

def complex_to_real_displacements(mu, hbar=2):
    r"""Returns the vector of complex displacements and conjugate displacements.
    Args:
        mu (array): length-:math:`2N` means vector
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the expectation values
        :math:`[\langle a_1\rangle, \langle a_2\rangle,\dots,\langle a_N\rangle, \langle a^\dagger_1\rangle, \dots, \langle a^\dagger_N\rangle]`
    """
    N = len(mu) // 2
    # mean displacement of each mode
    alpha = (mu[:N] + 1j * mu[N:]) / np.sqrt(2 * hbar)
    # the expectation values (<a_1>, <a_2>,...,<a_N>, <a^\dagger_1>, ..., <a^\dagger_N>)
    return np.concatenate([alpha, alpha.conj()])

def reduction(A, rpt):
    r"""Calculates the reduction of an array by a vector of indices.
    This is equivalent to repeating the ith row/column of :math:`A`, :math:`rpt_i` times.
    Args:
        A (array): matrix of size [N, N]
        rpt (Sequence): sequence of N positive integers indicating the corresponding rows/columns
            of A to be repeated.
    Returns:
        array: the reduction of A by the index vector rpt
    """
    rows = [i for sublist in [[idx] * j for idx, j in enumerate(rpt)] for i in sublist]

    if A.ndim == 1:
        return A[rows]

    return A[:, rows][rows]

def Xmat(N):
    r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}`
    Args:
        N (int): positive integer
    Returns:
        array: :math:`2N\times 2N` array
    """
    I = np.identity(N)
    O = np.zeros_like(I)
    X = np.block([[O, I], [I, O]])
    return X

def Qmat(cov, hbar=2):
    r"""Returns the :math:`Q` Husimi matrix of the Gaussian state.
    Args:
        cov (array): :math:`2N\times 2N xp-` Wigner covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
    Returns:
        array: the :math:`Q` matrix.
    """
    # number of modes
    N = len(cov) // 2
    I = np.identity(N)

    x = cov[:N, :N] * 2 / hbar
    xp = cov[:N, N:] * 2 / hbar
    p = cov[N:, N:] * 2 / hbar
    # the (Hermitian) matrix elements <a_i^\dagger a_j>
    aidaj = (x + p + 1j * (xp - xp.T) - 2 * I) / 4
    # the (symmetric) matrix elements <a_i a_j>
    aiaj = (x - p + 1j * (xp + xp.T)) / 4

    # calculate the covariance matrix sigma_Q appearing in the Q function:
    # Q(alpha) = exp[-(alpha-beta).sigma_Q^{-1}.(alpha-beta)/2]/|sigma_Q|
    Q = np.block([[aidaj, aiaj.conj()], [aiaj, aidaj.conj()]]) + np.identity(2 * N)
    return Q

def Amat(cov, hbar=2, cov_is_qmat=False):
    r"""Returns the :math:`A` matrix of the Gaussian state whose hafnian gives the photon number probabilities.
    Args:
        cov (array): :math:`2N\times 2N` covariance matrix
        hbar (float): the value of :math:`\hbar` in the commutation
            relation :math:`[\x,\p]=i\hbar`.
        cov_is_qmat (bool): if ``True``, it is assumed that ``cov`` is in fact the Q matrix.
    Returns:
        array: the :math:`A` matrix.
    """
    # number of modes
    N = len(cov) // 2
    X = Xmat(N)
    # inverse Q matrix
    if cov_is_qmat:
        Q = cov
    else:
        Q = Qmat(cov, hbar=hbar)
    Qinv = np.linalg.inv(Q)
    # calculate Hamilton's A matrix: A = X.(I-Q^{-1})*
    A = X @ (np.identity(2 * N) - Qinv).conj()
    return A

def _prefactor(mu, cov, hbar=2):
    r"""Returns the prefactor.
    .. math:: prefactor = \frac{e^{-\beta Q^{-1}\beta^*/2}}{n_1!\cdots n_m! \sqrt{|Q|}}
    Args:
        mu (array): length-:math:`2N` vector of mean values :math:`[\alpha,\alpha^*]`
        cov (array): length-:math:`2N` `xp`-covariance matrix
    Returns:
        float: the prefactor
    """
    Q = Qmat(cov, hbar=hbar)
    beta = complex_to_real_displacements(mu, hbar=hbar)
    Qinv = np.linalg.inv(Q)
    return np.exp(-0.5 * beta @ Qinv @ beta.conj()) / np.sqrt(np.linalg.det(Q))