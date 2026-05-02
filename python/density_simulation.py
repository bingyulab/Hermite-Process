"""
Rosenblatt Distribution Density Simulation
=============================================

Two algorithms for computing the density/CDF of the Rosenblatt distribution:

  Algorithm I  — Veillette & Taqqu (2013):
      Nyström eigenvalue computation (non-uniform mesh, analytical integrals)
      + direct FFT density from those eigenvalues
      + optional convolution F_{Z_D} = F_{X_M} * f_{Y_M} with Edgeworth tail.

  Algorithm II — Leonenko & Pepelyshev (2025):
      Closed-form eigenvalue approximation + direct Fourier inversion
      of the characteristic function via FFT.

Validation experiments compare the two methods and check against Monte Carlo
simulation of the double Wiener–Itô integral (from simulation.py).

References:
  - Veillette & Taqqu (2013), Properties and numerical evaluation of the
    Rosenblatt distribution
  - Leonenko & Pepelyshev (2025), Numerical computation of the Rosenblatt
    distribution
  - Tudor (2013), Analysis of Variations for Self-similar Processes
"""

import os
import math
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn
from scipy.integrate import quad
from scipy.stats import norm

# ════════════════════════════════════════════════════════════════════════════
# Logging setup
# ════════════════════════════════════════════════════════════════════════════
os.makedirs("../output/density/", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../output/density/density_simulation.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Shared eigenvalue utilities
# ════════════════════════════════════════════════════════════════════════════

def eigenvalue_first(a):
    """
    First eigenvalue (empirical formula from LP2025 / VT2013).
    λ_{a,1} = (1 + 0.1409 a) √(π^a Γ(1-a)) √(1/2 - a)
    """
    return (1.0 + 0.1409 * a) * np.sqrt(
        gamma_fn(1.0 - a) * np.pi ** a
    ) * np.sqrt(0.5 - a)


def eigenvalues_LP(a, K):
    """
    Closed-form eigenvalue approximation (Leonenko & Pepelyshev 2025).

    Parameters
    ----------
    a : float   Shape parameter in (0, 1/2).  a = 1 - H.
    K : int     Number of eigenvalues.

    Returns
    -------
    lam : ndarray, shape (K,)
        Raw (un-normalised) eigenvalues.  Their 2nd-power sum approaches
        1/2 as K → ∞.

    Note
    ────
    Use `eigenvalues_LP_normalised` to obtain unit-variance eigenvalues
    for sampling.  Use these raw values only for theoretical analysis or
    density computation via the characteristic function.
    """
    S   = np.sqrt((1.0 - 2.0 * a) * (1.0 - a) / 2.0)
    C_a = (
        2.0 * S * gamma_fn(1.0 - a) * np.sin(np.pi * a / 2.0)
        / np.pi ** (1.0 - a)
    )
    SecondCoef = (
        1.05
        * a ** (5.0 / 4.0)
        * np.sqrt(max(gamma_fn(1.0 - (0.5 - a)) - 1.0, 0.0))
    )
    n   = np.arange(1, K + 1, dtype=float)
    lam = C_a * n ** (a - 1.0) + SecondCoef * n ** (a - 2.2)
    lam[0] = eigenvalue_first(a)
    return np.maximum(lam, 0.0)

def eigenvalues_LP_normalised(a: float, K: int) -> np.ndarray:
    """
    FIX 1 – Unit-variance LP eigenvalues for Monte Carlo sampling.

    Rescales `eigenvalues_LP(a, K)` so that 2 * sum(lam^2) == 1 exactly,
    guaranteeing Var(Z_D) = 1 for any finite truncation K.

    Without this rescaling, sampling via
        Z = Σ_n λ_n (ξ_n^2 - 1)
    gives Var(Z) = 2 * sum(λ_n^2) < 1 for finite K, making the
    empirical variance of the noise slightly below unity.  This biases
    the cold-diffusion forward process: the nominal noise level σ(t)
    underestimates the actual corruption level, shifting training and
    test distributions apart.

    Parameters
    ----------
    a : float   a = 1 - H ∈ (0, 0.5).
    K : int     Number of eigenvalues.

    Returns
    -------
    lam_normed : ndarray, shape (K,)
        Eigenvalues scaled so that 2 * sum(lam_normed^2) == 1.
    """
    lam = eigenvalues_LP(a, K)
    var = 2.0 * np.sum(lam ** 2)
    if var > 0:
        lam = lam / np.sqrt(var)
    return lam

# ════════════════════════════════════════════════════════════════════════════
# Inline Monte-Carlo sampler (replaces broken `RosenblattSimulator` import)
# ════════════════════════════════════════════════════════════════════════════

def sample_rosenblatt_mc(a: float, K: int = 200,
                          n_samples: int = 30_000,
                          normalised: bool = True,
                          rng: np.random.Generator = None) -> np.ndarray:
    """
    FIX 2 – Inline MC sampler for the Rosenblatt distribution.

    Generates `n_samples` i.i.d. draws from Z_D via the spectral sum:
        Z = Σ_{n=1}^K λ_n (ξ_n^2 - 1),   ξ_n ~ N(0,1) i.i.d.

    Parameters
    ----------
    a         : float.  Shape parameter a = 1 - H ∈ (0, 0.5).
    K         : int.    Number of spectral terms (default 200).
    n_samples : int.    Number of samples.
    normalised: bool.   If True use unit-variance eigenvalues (recommended).
    rng       : optional NumPy Generator for reproducibility.

    Returns
    -------
    samples : ndarray, shape (n_samples,)
    """
    if rng is None:
        rng = np.random.default_rng()

    lam = eigenvalues_LP_normalised(a, K) if normalised else eigenvalues_LP(a, K)
    xi  = rng.standard_normal(size=(n_samples, K))      # (N, K)
    z   = (xi ** 2 - 1.0) @ lam                         # (N,)
    return z


# ════════════════════════════════════════════════════════════════════════════
# FFT-based Fourier inversion (shared by both algorithms)
# ════════════════════════════════════════════════════════════════════════════

def _density_from_chf(chf_func, x_min=-5.0, x_max=8.0,
                      N=2 ** 16, z_max=40.0):
    """
    Compute density on a uniform x-grid using inverse FFT of a
    characteristic function.

        f(x) = 1/(2 pi) int phi(z) exp(-i z x) dz

    Parameters
    ----------
    chf_func : callable
        chf_func(z_array) -> complex array.
    x_min, x_max : float
        Desired range for the output x-grid.
    N : int
        FFT length (power of 2 preferred).
    z_max : float
        Frequency cutoff:  z in [-z_max, z_max].

    Returns
    -------
    x_grid, density : ndarrays
    """
    dz = 2.0 * z_max / N
    j_arr = np.arange(N)
    z = -z_max + j_arr * dz          # z_j = -z_max + j * dz

    chf_vals = chf_func(z)

    dx = 2.0 * np.pi / (N * dz)
    x0 = -N / 2 * dx                 # centre x-grid around 0

    # Phase-shift for FFT:
    #   f(x_k) = (dz/2pi) * exp(i z_max x0) * exp(i z_max k dx)
    #            * FFT[ phi(z_j) * exp(-i j dz x0) ][k]
    g = chf_vals * np.exp(-1j * j_arr * dz * x0)
    F = np.fft.fft(g)

    k_arr = np.arange(N)
    phase = np.exp(1j * z_max * x0) * np.exp(1j * z_max * k_arr * dx)
    density_raw = np.real(phase * F) * dz / (2.0 * np.pi)

    x_grid = x0 + k_arr * dx

    mask = (x_grid >= x_min) & (x_grid <= x_max)
    x_out = x_grid[mask]
    d_out = np.maximum(density_raw[mask], 0.0)
    return x_out, d_out


# ════════════════════════════════════════════════════════════════════════════
# Algorithm II — Leonenko & Pepelyshev (2025)
# ════════════════════════════════════════════════════════════════════════════

class RosenblattDensityLP:
    """
    Density of the Rosenblatt distribution via the LP algorithm.

    FIX : `sigma_eps2` is now computed from the RAW (un-normalised)
    eigenvalues, which is correct for the theoretical characteristic
    function.  The normalised eigenvalues are stored separately as
    `eigenvalues_normed` for use in MC sampling.
    """

    def __init__(self, a=0.3, K=200):
        assert 0.0 < a < 0.5, "a must be in (0, 0.5)"
        self.a = a
        self.H = 1.0 - a
        self.K = K
        # Raw eigenvalues for characteristic function / density computation
        self.eigenvalues       = eigenvalues_LP(a, K)
        # FIX 3: residual Gaussian variance from the theoretical (raw) sum
        self.sigma_eps2        = max(0.0, 1.0 - 2.0 * np.sum(self.eigenvalues ** 2))

        # FIX : normalised eigenvalues for unit-variance MC sampling
        self.eigenvalues_normed = eigenvalues_LP_normalised(a, K)

    # ---- characteristic function ----------------------------

    def characteristic_function(self, z):
        """
        phi(z) = exp(-z^2 sigma_eps^2 / 2)
                 * prod_{n=1}^K exp( -1/2 ln(1 - 2i lam_n z) - i lam_n z )
        φ(z) = exp(-z² σ_ε² / 2)
               × Π_n exp( -½ ln(1 - 2i λ_n z) - i λ_n z )

        Uses raw (un-normalised) eigenvalues so that the theoretical
        variance E[Z²] = 1 - σ_ε² + 2*sum(λ_n²) = 1 is preserved.
        """
        z   = np.asarray(z, dtype=complex)
        chf = np.exp(-z ** 2 * self.sigma_eps2 / 2.0)
        for lam_n in self.eigenvalues:
            chf *= np.exp(
                -0.5 * np.log(1.0 - 2j * lam_n * z) - 1j * lam_n * z
            )
        return chf    

    # ---- density via FFT (fast) -----------------------------

    def density_fft(self, x_min=-5.0, x_max=8.0, N=2 ** 16, z_max=40.0):
        return _density_from_chf(
            self.characteristic_function,
            x_min=x_min, x_max=x_max, N=N, z_max=z_max,
        )

    # ---- density via scipy quad (reference) -----------------

    def density_quad(self, x_grid, z_max=20.0):
        """
        Gold-standard reference: integrate via quad.
        f(x) = 1/(2 pi) int_{-z_max}^{z_max} Re[ phi(z) exp(-i z x) ] dz
        """
        x_grid = np.asarray(x_grid, dtype=float)
        result = np.zeros_like(x_grid)
        for idx, x in enumerate(x_grid):
            def integrand(z, x=x):
                val = self.characteristic_function(np.array([z]))[0]
                return np.real(val * np.exp(-1j * z * x))
            val, _ = quad(integrand, -z_max, z_max, limit=500)
            result[idx] = val / (2.0 * np.pi)
        return result   

    # ------------------------------------------------------------------
    def sample(self, n_samples: int = 10_000,
               rng: np.random.Generator = None) -> np.ndarray:
        """
        Draw n_samples from Z_D using normalised eigenvalues.
        Convenience wrapper around `sample_rosenblatt_mc`.
        """
        return sample_rosenblatt_mc(
            self.a, K=self.K, n_samples=n_samples,
            normalised=True, rng=rng)

# ════════════════════════════════════════════════════════════════════════════
# Algorithm I — Veillette & Taqqu (2013)
# ════════════════════════════════════════════════════════════════════════════

class RosenblattDensityVT:
    """
    Density / CDF of the Rosenblatt distribution via VT.

    Faithfully follows the MATLAB reference code of Veillette & Taqqu:
      - Non-uniform mesh (meshgen.m) concentrating points near 0 and 1.
      - Analytical Nystrom matrix entries (closed-form integrals of
        |x_i - u|^{-D} against piecewise-linear basis functions).
    """

    def __init__(self, D=0.3, M0=50, N_grid=1500):
        assert 0.0 < D < 0.5, "D must be in (0, 0.5)"
        self.D = D
        self.H = 1.0 - D
        self.M0 = M0
        self.N_grid = N_grid

        self.sigma_D = np.sqrt(0.5 * (1.0 - 2.0 * D) * (1.0 - D))
        self.C_D = (
            2.0 / np.pi ** (1.0 - D)
        ) * self.sigma_D * gamma_fn(1.0 - D) * np.sin(np.pi * D / 2.0)

        log.info(
            "VT: computing eigenvalues (D=%.4f, N_grid=%d, M0=%d) ...",
            D, N_grid, M0,
        )
        t0 = time.time()
        self.eigenvalues = self._compute_eigenvalues_nystrom()
        log.info(
            "VT: done in %.2f s  (M=%d, max_eig=%.6f, sum_lam2=%.6f)",
            time.time() - t0,
            len(self.eigenvalues),
            self.eigenvalues[0],
            np.sum(self.eigenvalues ** 2),
        )

    # ---- Non-uniform mesh (MATLAB meshgen.m) ----------------

    @staticmethod
    def _meshgen(N):
        """
        Non-uniform mesh on [0, 1] with N points.
        Concentrates grid points near 0 and 1 where the kernel
        |x-u|^{-D} is most singular.

        Matches MATLAB meshgen.m exactly:
            x = linspace(0,1,N)
            y(x <= 0.5) = x.^4 * 0.5^(-3)
            y(x > 0.5)  = 1 - (1-x).^4 * 0.5^(-3)
        """
        x = np.linspace(0.0, 1.0, N)
        y = np.empty_like(x)
        lo = x <= 0.5
        hi = ~lo
        y[lo] = x[lo] ** 4 * 0.5 ** (-3)
        y[hi] = 1.0 - (1.0 - x[hi]) ** 4 * 0.5 ** (-3)
        return y

    # ---- Nystrom matrix (MATLAB RosenblattEigs.m) -----------

    def _build_nystrom_matrix(self, N):
        """
        Build the N x N Nystrom matrix using ANALYTICAL integration of
        |x_i - u|^{-D} against piecewise-linear basis functions on the
        non-uniform mesh.

        This matches MATLAB RosenblattEigs.m exactly.
        """
        D = self.D
        x = self._meshgen(N)
        e = np.diff(x)                 # segment lengths, shape (N-1,)

        D1 = 1.0 - D
        D2 = 2.0 - D

        T = np.zeros((N, N))

        for i in range(N):
            xi = x[i]

            # ---- segments j < i  (u < x_i, so |x_i-u| = x_i-u) ----
            for j in range(i):
                ej = e[j]
                a1 = xi - x[j]        # > 0
                a2 = xi - x[j + 1]    # >= 0

                # left  hat (x_{j+1} - u) / e_j
                c = a1 ** D1 / D1 - (a1 ** D2 - a2 ** D2) / (D1 * D2 * ej)
                # right hat (u - x_j) / e_j
                d = (a1 ** D2 - a2 ** D2) / (D1 * D2 * ej) - a2 ** D1 / D1

                T[i, j] += c
                T[i, j + 1] += d

            # ---- segments j >= i (u >= x_i, so |x_i-u| = u-x_i) ----
            for j in range(i, N - 1):
                ej = e[j]
                b1 = x[j + 1] - xi    # > 0
                b2 = x[j] - xi        # >= 0

                # left  hat (x_{j+1} - u) / e_j
                c = (b1 ** D2 - b2 ** D2) / (D1 * D2 * ej) - b2 ** D1 / D1
                # right hat (u - x_j) / e_j
                d = b1 ** D1 / D1 - (b1 ** D2 - b2 ** D2) / (D1 * D2 * ej)

                T[i, j] += c
                T[i, j + 1] += d

        return T

    def _compute_eigenvalues_nystrom(self):
        """
        Compute the M0 largest eigenvalues of sigma_D * K_D
        via the Nystrom method.

        IMPORTANT: On a non-uniform mesh, the collocation matrix T is
        NOT symmetric (hat functions have different widths on each side).
        Symmetrising T and using eigsh would corrupt the eigenvalues.
        We must use np.linalg.eig on the non-symmetric matrix.
        """
        T = self._build_nystrom_matrix(self.N_grid)
        T_scaled = self.sigma_D * T

        # Use general (non-symmetric) eigenvalue solver
        all_eigs = np.linalg.eig(T_scaled)[0]

        # True eigenvalues of the integral operator are real and positive.
        # Discard eigenvalues with significant imaginary parts.
        max_imag = np.max(np.abs(all_eigs.imag))
        if max_imag > 1e-6:
            log.warning("VT: max |Im(eig)| = %.2e (expected ~0)", max_imag)

        # Take real parts, keep positive, sort descending
        real_eigs = all_eigs.real
        real_eigs = real_eigs[real_eigs > 1e-12]
        real_eigs = np.sort(real_eigs)[::-1]

        # Return top M0
        return real_eigs[: self.M0]

    # ---- Characteristic function (from Nystrom eigenvalues) --

    def characteristic_function(self, z):
        """
        phi(z) = exp(-z^2 sigma^2 / 2)
                 * prod exp( -1/2 ln(1 - 2i lam z) - i lam z )
        """
        sigma2 = max(0.0, 1.0 - 2.0 * np.sum(self.eigenvalues ** 2))
        z = np.asarray(z, dtype=complex)
        chf = np.exp(-z ** 2 * sigma2 / 2.0)
        for lam_n in self.eigenvalues:
            chf *= np.exp(
                -0.5 * np.log(1.0 - 2j * lam_n * z) - 1j * lam_n * z
            )
        return chf

    # ---- density via FFT ------------------------------------

    def density_fft_direct(self, x_min=-5.0, x_max=8.0,
                           N_fft=2 ** 16, z_max=40.0):
        return _density_from_chf(
            self.characteristic_function,
            x_min=x_min, x_max=x_max, N=N_fft, z_max=z_max,
        )

    # ---- density via quad (reference, slow) -----------------

    def density_quad(self, x_grid, z_max=20.0):
        sigma2 = max(0.0, 1.0 - 2.0 * np.sum(self.eigenvalues ** 2))
        x_grid = np.asarray(x_grid, dtype=float)
        result = np.zeros_like(x_grid)

        for idx, xv in enumerate(x_grid):
            def integrand(z, xv=xv):
                chf = np.exp(-z ** 2 * sigma2 / 2.0 + 0j)
                for lam_n in self.eigenvalues:
                    chf *= np.exp(
                        -0.5 * np.log(1.0 - 2j * lam_n * z)
                        - 1j * lam_n * z
                    )
                return np.real(chf * np.exp(-1j * z * xv))

            val, _ = quad(integrand, -z_max, z_max, limit=500)
            result[idx] = val / (2.0 * np.pi)
        return result

    # ---- Edgeworth convolution (full VT pipeline) -----------

    @staticmethod
    def _hurwitz_zeta(s, M):
        """Hurwitz zeta:  zeta(s, M) = sum_{n=M}^inf n^{-s}"""
        from scipy.special import zeta as riemann_zeta
        zeta_s = float(riemann_zeta(s))
        head = np.sum(np.arange(1, M, dtype=float) ** (-s)) if M > 1 else 0.0
        return zeta_s - head

    @staticmethod
    def _hermite_prob(n, x):
        """Probabilist's Hermite polynomial He_n(x) via recurrence."""
        x = np.asarray(x, dtype=float)
        if n == 0:
            return np.ones_like(x)
        if n == 1:
            return x.copy()
        H_prev = np.ones_like(x)
        H_curr = x.copy()
        for k in range(2, n + 1):
            H_next = x * H_curr - (k - 1) * H_prev
            H_prev, H_curr = H_curr, H_next
        return H_curr

    @staticmethod
    def _eta_tuples(N):
        """Edgeworth expansion index tuples, matching MATLAB eta.m."""
        tuples_dict = {
            3: [[1]],
            4: [[2, 0], [0, 1]],
            5: [[3, 0, 0], [1, 1, 0], [0, 0, 1]],
            6: [[4, 0, 0, 0], [2, 1, 0, 0], [0, 2, 0, 0],
                [1, 0, 1, 0], [0, 0, 0, 1]],
            7: [[5, 0, 0, 0, 0], [3, 1, 0, 0, 0], [1, 2, 0, 0, 0],
                [2, 0, 1, 0, 0], [0, 1, 1, 0, 0], [1, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]],
            8: [[6, 0, 0, 0, 0, 0], [4, 1, 0, 0, 0, 0],
                [2, 2, 0, 0, 0, 0], [0, 3, 0, 0, 0, 0],
                [3, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0],
                [0, 0, 2, 0, 0, 0], [2, 0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]],
        }
        if N < 3:
            return []
        return tuples_dict.get(min(N, 8), [])

    def density_convolution(self, z_grid, M=None, N_edge=5):
        """
        Full VT convolution:
            f_{Z_D}(z) = int f_{X_M}(z - y) f_{Y_M}(y) dy

        When sigma_M is negligible, the tail Y_M contributes nothing
        and we fall back to the X_M density directly.
        """
        if M is None:
            M = min(len(self.eigenvalues), 50)

        evals = self.eigenvalues[: M - 1]
        sigma_M2 = 1.0 - 2.0 * np.sum(evals ** 2)
        sigma_M = np.sqrt(max(0.0, sigma_M2))

        log.info("  Convolution: M=%d, #evals=%d, sigma_M=%.6f",
                 M, len(evals), sigma_M)

        def pdf_X_M(xv):
            """PDF of X_M = sum_{n=1}^{M-1} lam_n (eps_n^2 - 1) via Fourier inversion."""
            def integrand(t, xv=xv):
                chf = 1.0 + 0j
                for lam in evals:
                    chf *= np.exp(-1j * t * lam) / np.sqrt(
                        1.0 - 2j * t * lam
                    )
                return np.real(chf * np.exp(-1j * t * xv))
            val, _ = quad(integrand, -100, 100, limit=500)
            return val / (2.0 * np.pi)

        z_grid = np.asarray(z_grid, dtype=float)
        pdf_out = np.zeros_like(z_grid)

        # If sigma_M is negligible, Y_M contributes nothing:
        # f_{Z_D}(z) ≈ f_{X_M}(z)
        if sigma_M < 1e-8:
            log.info("  sigma_M ≈ 0: using X_M density directly (no Edgeworth tail)")
            for i, z_val in enumerate(z_grid):
                pdf_out[i] = max(0.0, pdf_X_M(z_val))
            return pdf_out

        # Normalised tail cumulants kap_{k,M} of Y_M / sigma_M
        kapM = np.zeros(N_edge + 1)
        kapM[2] = 1.0
        for k in range(3, N_edge + 1):
            tail_sum = self.C_D ** k * self._hurwitz_zeta(
                k * (1.0 - self.D), M
            )
            kapM[k] = (
                2 ** (k - 1) * math.factorial(k - 1) * tail_sum
                / sigma_M ** k
            )

        eta_N = self._eta_tuples(N_edge)

        def edgeworth_pdf(u):
            """PDF of Y_M / sigma_M with Edgeworth correction."""
            phi_u = norm.pdf(u)
            correction = 1.0
            for kvec in eta_N:
                m_idx = np.arange(3, 3 + len(kvec))
                zeta_p1 = int(np.sum(m_idx * np.array(kvec)))
                coeff = 1.0
                for ii, ki in enumerate(kvec):
                    m = m_idx[ii]
                    coeff *= (
                        1.0 / math.factorial(ki)
                    ) * (kapM[m] / math.factorial(m)) ** ki
                correction += coeff * self._hermite_prob(zeta_p1, u)
            return phi_u * correction

        # Convolution: f_{Z_D}(z) = int f_{X_M}(z - y) * f_{Y_M}(y) dy
        for i, z_val in enumerate(z_grid):
            def conv_integrand(y, z_val=z_val):
                fXM = pdf_X_M(z_val - y)
                fYM = edgeworth_pdf(y / sigma_M) / sigma_M
                return fXM * fYM

            y_lo = -6.0 * sigma_M
            y_hi = 6.0 * sigma_M
            val, _ = quad(conv_integrand, y_lo, y_hi, limit=200)
            pdf_out[i] = max(0.0, val)

        return pdf_out


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _H_to_a(H): return 1.0 - H

def _a_to_H(a): return 1.0 - a


# ════════════════════════════════════════════════════════════════════════════
# Experiment 1: LP density — FFT vs quad (validation of FFT)
# ════════════════════════════════════════════════════════════════════════════

def experiment_fft_vs_quad():
    log.info("=" * 70)
    log.info("Experiment 1: LP density — FFT vs quad (FFT validation)")
    log.info("=" * 70)

    a_vals = [0.15, 0.25, 0.35, 0.45]
    K = 200

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, a in enumerate(a_vals):
        H = _a_to_H(a)
        log.info("--- a = %.2f  (H = %.2f) ---", a, H)

        lp = RosenblattDensityLP(a=a, K=K)

        t0 = time.time()
        x_fft, d_fft = lp.density_fft(
            x_min=-3.0, x_max=6.0, N=2 ** 16, z_max=40.0
        )
        dt_fft = time.time() - t0
        log.info("  FFT: %.3f s", dt_fft)

        x_quad = np.linspace(-3.0, 6.0, 80)
        t0 = time.time()
        # Use z_max=40 to match FFT range (z_max=20 truncates too early for small a)
        d_quad = lp.density_quad(x_quad, z_max=40.0)
        dt_quad = time.time() - t0
        log.info("  Quad: %.3f s", dt_quad)

        d_fft_interp = np.interp(x_quad, x_fft, d_fft)
        max_abs = np.max(np.abs(d_fft_interp - d_quad))
        mask = d_quad > 0.01
        max_rel = (
            np.max(np.abs(d_fft_interp[mask] - d_quad[mask]) / d_quad[mask])
            if np.any(mask)
            else np.nan
        )
        log.info(
            "  Max |FFT - quad|: %.6f, Max rel (d>0.01): %.6f",
            max_abs, max_rel,
        )

        ax = axes[idx]
        ax.plot(x_fft, d_fft, "b-", lw=2, label="FFT")
        ax.plot(x_quad, d_quad, "ro", ms=5, label="Quad")
        ax.set_title(f"a = {a:.2f}  (H = {H:.2f})")
        ax.set_xlabel("x"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Algorithm II (LP): FFT vs Quadrature Validation",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("../output/density/density_fft_vs_quad.png", dpi=150, bbox_inches="tight")
    log.info("Saved density_fft_vs_quad.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 2: Compare two algorithms
# ════════════════════════════════════════════════════════════════════════════

def experiment_compare_algorithms():
    log.info("=" * 70)
    log.info("Experiment 2: Compare Algorithm I (VT) vs Algorithm II (LP)")
    log.info("=" * 70)

    a_vals = [0.15, 0.25, 0.35, 0.45]
    K_lp = 200

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, a in enumerate(a_vals):
        H = _a_to_H(a)
        log.info("--- a = %.2f  (H = %.2f) ---", a, H)

        # Algorithm II (LP) — fast
        t0 = time.time()
        lp = RosenblattDensityLP(a=a, K=K_lp)
        x_lp, d_lp = lp.density_fft(
            x_min=-3.0, x_max=6.0, N=2 ** 16, z_max=40.0
        )
        dt_lp = time.time() - t0
        log.info("  LP (FFT, K=%d): %.3f s", K_lp, dt_lp)

        # Algorithm I (VT) — Nystrom eigenvalues + FFT
        t0 = time.time()
        vt = RosenblattDensityVT(D=a, M0=50, N_grid=1500)
        x_vt, d_vt = vt.density_fft_direct(
            x_min=-3.0, x_max=6.0, N_fft=2 ** 16, z_max=40.0
        )
        dt_vt = time.time() - t0
        log.info("  VT (Nystrom+FFT, M0=%d): %.3f s", vt.M0, dt_vt)

        ax = axes[idx]
        ax.plot(x_lp, d_lp, "b-", lw=2, label="LP (Alg II)")
        ax.plot(x_vt, d_vt, "r--", lw=2, label="VT (Alg I)")
        ax.set_title(f"a = {a:.2f}  (H = {H:.2f})", fontsize=13)
        ax.set_xlabel("x"); ax.set_ylabel("Density")
        ax.legend(); ax.grid(True, alpha=0.3)

        d_vt_i = np.interp(x_lp, x_vt, d_vt)
        max_abs = np.max(np.abs(d_lp - d_vt_i))
        mask = d_lp > 0.01
        max_rel = (
            np.max(np.abs(d_lp[mask] - d_vt_i[mask]) / d_lp[mask])
            if np.any(mask) else np.nan
        )
        log.info("  Max |LP-VT|: %.6f, Max rel (d>0.01): %.4f", max_abs, max_rel)
        log.info("  Speed ratio VT/LP: %.1fx", dt_vt / max(dt_lp, 1e-6))

    plt.suptitle(
        "Rosenblatt Density: Algorithm I (VT) vs Algorithm II (LP)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("../output/density/density_compare_algorithms.png", dpi=150, bbox_inches="tight")
    log.info("Saved density_compare_algorithms.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 3: Validate against Monte Carlo
# ════════════════════════════════════════════════════════════════════════════

def experiment_validate_mc():
    log.info("=" * 70)
    log.info("Experiment 3: Validate density against Monte Carlo simulation")
    log.info("=" * 70)

    H        = 0.75
    a        = _H_to_a(H)
    K        = 200
    n_samples = 30_000

    log.info("H=%.2f (a=%.2f), n_samples=%d, K=%d", H, a, n_samples, K)

    # ── MC samples ────────────────────────────────
    rng = np.random.default_rng(seed=42)
    t0  = time.time()
    samples = sample_rosenblatt_mc(a, K=K, n_samples=n_samples,
                                    normalised=True, rng=rng)
    dt_mc   = time.time() - t0
    log.info("  MC sampling (inline spectral): %.2f s", dt_mc)

    mu    = np.mean(samples)
    std_v = np.std(samples)
    skew  = float(np.mean(((samples - mu) / max(std_v, 1e-8)) ** 3))
    kurt  = float(np.mean(((samples - mu) / max(std_v, 1e-8)) ** 4))
    log.info("  MC: mean=%.4f std=%.4f skew=%.4f kurt=%.4f", mu, std_v, skew, kurt)

    # ── LP and VT densities ──────────────────────────────────────────────
    t0 = time.time()
    lp = RosenblattDensityLP(a=a, K=K)
    x_lp, d_lp = lp.density_fft(x_min=-3.0, x_max=6.0, N=2 ** 16)
    log.info("  LP density: %.3f s", time.time() - t0)

    t0 = time.time()
    vt = RosenblattDensityVT(D=a, M0=50, N_grid=1500)
    x_vt, d_vt = vt.density_fft_direct(x_min=-3.0, x_max=6.0, N_fft=2 ** 16)
    log.info("  VT density: %.3f s", time.time() - t0)

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples, bw_method="scott")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(samples, bins=120, density=True, alpha=0.3,
            color="steelblue", label="MC histogram")
    ax.plot(x_lp, d_lp, "b-",  lw=2, label="LP (Alg II, fixed)")
    ax.plot(x_vt, d_vt, "r--", lw=2, label="VT (Alg I)")
    x_kde = np.linspace(-3, 6, 300)
    ax.plot(x_kde, kde(x_kde), "g:", lw=1.5, label="MC KDE")
    ax.set_xlabel("x"); ax.set_ylabel("Density")
    ax.set_title(f"Rosenblatt Density Validation (H={H})")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    d_kde = kde(x_lp)
    mask  = d_kde > 0.01
    if np.any(mask):
        rel_lp = np.abs(d_lp[mask] - d_kde[mask]) / d_kde[mask]
        d_vt_i = np.interp(x_lp, x_vt, d_vt)
        rel_vt = np.abs(d_vt_i[mask] - d_kde[mask]) / d_kde[mask]
        ax.plot(x_lp[mask], rel_lp, "b-",  lw=1.5, label="|LP-KDE|/KDE")
        ax.plot(x_lp[mask], rel_vt, "r--", lw=1.5, label="|VT-KDE|/KDE")
        log.info("  Max rel err LP vs KDE: %.4f", np.max(rel_lp))
        log.info("  Max rel err VT vs KDE: %.4f", np.max(rel_vt))
    ax.set_xlabel("x"); ax.set_ylabel("Relative error")
    ax.set_title("Relative Error vs MC KDE")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/density/density_validate_mc.png", dpi=150, bbox_inches="tight")
    log.info("Saved density_validate_mc.png"); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 4: Eigenvalue comparison
# ════════════════════════════════════════════════════════════════════════════

def experiment_eigenvalue_comparison():
    log.info("=" * 70)
    log.info("Experiment 4: Eigenvalue comparison (Nystrom vs closed-form)")
    log.info("=" * 70)

    a_vals = [0.15, 0.25, 0.35, 0.45]
    M = 50

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, a in enumerate(a_vals):
        log.info("--- a = %.2f ---", a)

        lam_lp = eigenvalues_LP(a, M)
        vt = RosenblattDensityVT(D=a, M0=M, N_grid=1500)
        lam_vt = vt.eigenvalues[:M]

        n_arr = np.arange(1, M + 1)

        ax = axes[idx]
        ax.semilogy(n_arr, lam_lp, "bo-", ms=4, label="LP (closed-form)")
        ax.semilogy(n_arr, lam_vt, "r^--", ms=4, label="VT (Nystrom)")
        ax.set_xlabel("n"); ax.set_ylabel(r"$\lambda_n$")
        ax.set_title(f"a = {a:.2f}  (H = {1-a:.2f})")
        ax.legend(); ax.grid(True, alpha=0.3)

        rel = np.abs(lam_lp - lam_vt) / (lam_lp + 1e-15)
        log.info("  Max rel eig diff: %.4f", np.max(rel))
        log.info("  Mean rel eig diff: %.4f", np.mean(rel))
        log.info(
            "  sum_lam2 (LP K=%d): %.6f, sum_lam2 (VT M=%d): %.6f  (th: 0.5)",
            M, np.sum(lam_lp ** 2), M, np.sum(lam_vt ** 2),
        )

    plt.suptitle(
        "Eigenvalue Comparison: LP vs VT (Nystrom)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("../output/density/eigenvalue_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/density/eigenvalue_comparison.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 5: Speed benchmark
# ════════════════════════════════════════════════════════════════════════════

def experiment_speed_benchmark():
    log.info("=" * 70)
    log.info("Experiment 5: Speed benchmark")
    log.info("=" * 70)

    a = 0.25
    K_vals = [50, 100, 200, 500]
    lp_times, vt_times = [], []

    for K in K_vals:
        t0 = time.time()
        lp = RosenblattDensityLP(a=a, K=K)
        lp.density_fft(x_min=-3.0, x_max=6.0, N=2 ** 16)
        dt_lp = time.time() - t0
        lp_times.append(dt_lp)

        M0 = min(K, 50)
        t0 = time.time()
        vt = RosenblattDensityVT(D=a, M0=M0, N_grid=1500)
        vt.density_fft_direct(x_min=-3.0, x_max=6.0, N_fft=2 ** 16)
        dt_vt = time.time() - t0
        vt_times.append(dt_vt)

        log.info(
            "  K=%d: LP=%.3f s, VT=%.3f s (ratio=%.1fx)",
            K, dt_lp, dt_vt, dt_vt / max(dt_lp, 1e-6),
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_vals, lp_times, "bo-", lw=2, ms=8, label="LP (Alg II)")
    ax.plot(K_vals, vt_times, "r^--", lw=2, ms=8, label="VT (Alg I)")
    ax.set_xlabel("Number of eigenvalues K", fontsize=12)
    ax.set_ylabel("Wall-clock time (s)", fontsize=12)
    ax.set_title(f"Speed Benchmark (a = {a}, H = {1-a})", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/density/speed_benchmark.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/density/speed_benchmark.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 6: Density for multiple H values
# ════════════════════════════════════════════════════════════════════════════

def experiment_density_multiple_H():
    log.info("=" * 70)
    log.info("Experiment 6: Density for multiple H values")
    log.info("=" * 70)

    H_vals = [0.55, 0.65, 0.75, 0.85, 0.95]
    K = 200

    fig, ax = plt.subplots(figsize=(10, 6))
    for H in H_vals:
        a = _H_to_a(H)
        log.info("  H = %.2f  (a = %.2f)", H, a)
        lp = RosenblattDensityLP(a=a, K=K)
        x, d = lp.density_fft(x_min=-3.0, x_max=6.0, N=2 ** 16)
        ax.plot(x, d, lw=2, label=f"H = {H}")

    xg = np.linspace(-3, 6, 300)
    ax.plot(xg, norm.pdf(xg), "k--", lw=1.5, alpha=0.5, label="N(0,1)")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Rosenblatt Density for Various H (Algorithm II)", fontsize=14)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/density/density_multiple_H.png", dpi=150, bbox_inches="tight")
    log.info("Saved density_multiple_H.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 7: Cumulant / moment check
# ════════════════════════════════════════════════════════════════════════════

def experiment_cumulants():
    log.info("=" * 70)
    log.info("Experiment 7: Cumulant / moment check")
    log.info("=" * 70)

    a = 0.25
    K = 200

    lam_lp = eigenvalues_LP(a, K)
    vt = RosenblattDensityVT(D=a, M0=50, N_grid=1500)
    lam_vt = vt.eigenvalues

    log.info("a = %.2f, K_LP = %d, M_VT = %d", a, K, len(lam_vt))

    hdr = f"{'k':>4s}  {'kappa_k (LP)':>14s}  {'kappa_k (VT)':>14s}"
    log.info(hdr); log.info("-" * len(hdr))

    for k in range(2, 7):
        kl = 2 ** (k - 1) * math.factorial(k - 1) * np.sum(lam_lp ** k)
        kv = 2 ** (k - 1) * math.factorial(k - 1) * np.sum(lam_vt ** k)
        log.info("  %d   %14.6f  %14.6f", k, kl, kv)

    k2_lp = 2 * np.sum(lam_lp ** 2)
    k2_vt = 2 * np.sum(lam_vt ** 2)
    log.info("kappa_2 = 2 sum_lam2:  LP=%.6f, VT=%.6f  (th=1.0)", k2_lp, k2_vt)


# ════════════════════════════════════════════════════════════════════════════
# Experiment 8: VT convolution vs direct FFT
# ════════════════════════════════════════════════════════════════════════════

def experiment_vt_convolution_vs_direct():
    log.info("=" * 70)
    log.info("Experiment 8: VT convolution vs direct FFT")
    log.info("=" * 70)

    a = 0.3
    vt = RosenblattDensityVT(D=a, M0=50, N_grid=1500)

    t0 = time.time()
    x_dir, d_dir = vt.density_fft_direct(
        x_min=-3.0, x_max=6.0, N_fft=2 ** 16
    )
    dt_dir = time.time() - t0
    log.info("  Direct FFT: %.3f s", dt_dir)

    x_conv = np.linspace(-2.0, 5.0, 40)
    t0 = time.time()
    d_conv = vt.density_convolution(x_conv, M=20, N_edge=5)
    dt_conv = time.time() - t0
    log.info("  Convolution (M=20, 40 pts): %.3f s", dt_conv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(x_dir, d_dir, "b-", lw=2, label="Direct FFT")
    ax.plot(x_conv, d_conv, "ro", ms=6, label="Convolution (M=20)")
    ax.set_xlabel("x"); ax.set_ylabel("Density")
    ax.set_title(f"VT: Direct FFT vs Convolution (a={a})")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    d_di = np.interp(x_conv, x_dir, d_dir)
    diff = np.abs(d_di - d_conv)
    ax.plot(x_conv, diff, "k-o", lw=1.5, ms=4)
    ax.set_xlabel("x"); ax.set_ylabel("|Direct FFT - Convolution|")
    ax.set_title("Absolute Difference"); ax.grid(True, alpha=0.3)
    log.info("  Max |direct - conv|: %.6f", np.max(diff))

    plt.tight_layout()
    plt.savefig("../output/density/vt_conv_vs_direct.png", dpi=150, bbox_inches="tight")
    log.info("Saved vt_conv_vs_direct.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Experiment 9: Exponential bounds on density derivatives
# ════════════════════════════════════════════════════════════════════════════

def experiment_exponential_bounds():
    """
    Experiment 9: Verify exponential bounds on density derivatives.

    From Theorem 2 of Loosveldt & Pène (2025):
        |∂^n p(x)| ≤ C · t^{-nH} · exp(-c|x|)

    For a single time point t=1, this simplifies to:
        |p^{(n)}(x)| ≤ C_n · exp(-c|x|)

    We compute numerical derivatives and compare against the bound.
    """
    log.info("=" * 70)
    log.info("Experiment 9: Exponential bounds on density derivatives")
    log.info("(Theorem 2 of Loosveldt & Pène 2025)")
    log.info("=" * 70)

    a = 0.3   # shape parameter; H = 1 - a = 0.7
    H = 1.0 - a

    # Use LP algorithm for high-quality density
    lp = RosenblattDensityLP(a=a, K=300)

    # Compute density on a fine grid
    x_min, x_max = -4.0, 8.0
    x_grid, density = lp.density_fft(x_min=x_min, x_max=x_max, N=2**18, z_max=60.0)
    dx = x_grid[1] - x_grid[0]

    log.info("  Grid: %d points, dx=%.6f, range [%.1f, %.1f]",
             len(x_grid), dx, x_min, x_max)

    # Compute numerical derivatives using central differences
    # d^n f / dx^n ≈ (finite difference formula)
    def numerical_derivative(f, n, dx):
        """Compute n-th derivative using finite differences."""
        result = f.copy()
        for _ in range(n):
            # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / 2h
            result = np.gradient(result, dx)
        return result

    derivatives = {n: numerical_derivative(density, n, dx) for n in range(4)}

    # Fit exponential bounds: |p^{(n)}(x)| ≤ C_n exp(-c_n |x|)
    # For tails (|x| > threshold), fit log|p^{(n)}| ≈ log(C_n) - c_n |x|
    def fit_exponential_bound(x, y_abs, tail_threshold=1.5):
        """Fit C exp(-c|x|) to |y| in the tail region."""
        mask = np.abs(x) > tail_threshold
        if np.sum(mask) < 10:
            return np.nan, np.nan

        x_tail = np.abs(x[mask])
        y_tail = y_abs[mask]

        # Filter out zeros/negatives for log
        valid = y_tail > 1e-15
        if np.sum(valid) < 5:
            return np.nan, np.nan

        x_fit = x_tail[valid]
        log_y = np.log(y_tail[valid])

        # Linear regression: log|y| = log(C) - c|x|
        A = np.vstack([np.ones_like(x_fit), -x_fit]).T
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
            log_C, c = coeffs
            C = np.exp(log_C)
            return C, c
        except:
            return np.nan, np.nan

    # Fit bounds for each derivative order
    bounds = {}
    log.info("")
    log.info("  Fitted exponential bounds |p^{(n)}(x)| ≤ C_n exp(-c_n |x|):")
    log.info("  " + "-" * 50)
    log.info("  %5s  %12s  %12s  %15s", "n", "C_n", "c_n", "Bound tight?")
    log.info("  " + "-" * 50)

    for n in range(4):
        y_abs = np.abs(derivatives[n])
        C_n, c_n = fit_exponential_bound(x_grid, y_abs, tail_threshold=2.0)
        bounds[n] = (C_n, c_n)

        # Check tightness: ratio of max actual to bound
        if not np.isnan(C_n) and not np.isnan(c_n):
            bound_vals = C_n * np.exp(-c_n * np.abs(x_grid))
            max_ratio = np.max(y_abs / (bound_vals + 1e-30))
            tight = "Yes" if max_ratio < 2.0 else "Moderate" if max_ratio < 10 else "Loose"
            log.info("  %5d  %12.4e  %12.4f  %15s", n, C_n, c_n, tight)
        else:
            log.info("  %5d  %12s  %12s  %15s", n, "N/A", "N/A", "Fit failed")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'green', 'orange', 'red']
    labels = ['p(x)', "p'(x)", "p''(x)", "p'''(x)"]

    for n, ax in enumerate(axes.flat):
        y_abs = np.abs(derivatives[n])
        C_n, c_n = bounds[n]

        # Plot |derivative| on log scale
        ax.semilogy(x_grid, y_abs, '-', color=colors[n], lw=1.5,
                    label=f'|{labels[n]}| (numerical)')

        # Plot fitted exponential bound
        if not np.isnan(C_n) and not np.isnan(c_n):
            bound = C_n * np.exp(-c_n * np.abs(x_grid))
            ax.semilogy(x_grid, bound, 'k--', lw=2,
                        label=f'Bound: {C_n:.2e} exp(-{c_n:.3f}|x|)')

        # Also plot a universal bound with c from Wiener chaos tail estimate
        # From Lemma 1 of the paper: P(|X|/sqrt(E[X^2]) ≥ t) ≤ exp(-c_q t^{2/q})
        # For q=2: c_2 is a universal constant; tails decay like exp(-c sqrt(x))
        # But the density derivative bound uses linear decay exp(-c|x|)

        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel(f'|{labels[n]}|', fontsize=11)
        ax.set_title(f'Derivative order n={n}', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([1e-15, None])

    plt.suptitle(
        f"Exponential Bounds on Rosenblatt Density Derivatives (H={H}, a={a})\n"
        "Theorem 2 (Loosveldt & Pène 2025): $|\\partial^n p(x)| \\leq C \\exp(-c|x|)$",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("../output/density/density_exponential_bounds.png", dpi=150, bbox_inches="tight")
    log.info("Saved density_exponential_bounds.png")
    plt.close()

    # Additional plot: overlay all derivatives to show decay rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    for n in range(4):
        y_abs = np.abs(derivatives[n])
        ax.semilogy(x_grid, y_abs, '-', color=colors[n], lw=1.5, label=labels[n])

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('|Derivative|', fontsize=12)
    ax.set_title(f'Comparison of Derivative Decay Rates (H={H})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([1e-12, 10])

    plt.tight_layout()
    plt.savefig("../output/density/density_derivative_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved density_derivative_comparison.png")
    plt.close()

    # Report on tightness
    log.info("")
    log.info("  Analysis of bound tightness:")
    log.info("  " + "-" * 60)

    # Compare decay rates c_n across derivative orders
    c_values = [bounds[n][1] for n in range(4) if not np.isnan(bounds[n][1])]
    if len(c_values) > 1:
        log.info("  Decay rates c_n: %s", ", ".join([f"c_{n}={bounds[n][1]:.4f}"
                                                      for n in range(4)
                                                      if not np.isnan(bounds[n][1])]))
        log.info("  Mean decay rate: c = %.4f", np.mean(c_values))
        log.info("  The theorem predicts a UNIVERSAL c > 0 independent of n.")
        if np.std(c_values) < 0.1 * np.mean(c_values):
            log.info("  → GOOD: Decay rates are consistent (std/mean = %.2f%%)",
                     100 * np.std(c_values) / np.mean(c_values))
        else:
            log.info("  → Note: Some variation in decay rates observed")

    return derivatives, bounds


# ════════════════════════════════════════════════════════════════════════════
# Experiment 10 : Eigenvalue normalisation — bias quantification
# ════════════════════════════════════════════════════════════════════════════

def experiment_density_discrepancy():
    """
    Quantifies the bias introduced by the original cold-diffusion eigenvalue
    formula vs the reference LP formula.

    The original `Rosenblatt_cold_diffusion.py` computed:
        C_D = 2σ_D / (π^(1-D) · Γ(1-D) · sin(πD/2))
    which differs from the LP reference:
        C_a = 2σ_D · Γ(1-a) · sin(πa/2) / π^(1-a)

    After the variance rescaling in both files, the RELATIVE shape of the
    eigenvalue sequences still differs (because the second-order correction
    term also differs).  This changes the skewness and kurtosis of the
    sampled noise, which in turn shifts the training distribution.

    This experiment plots:
      (a) The raw eigenvalue sequences from both formulas.
      (b) The resulting density functions.
      (c) The cumulant differences.
    """
    log.info("=" * 70)
    log.info("Experiment 10 (NEW): Eigenvalue normalisation — bias quantification")
    log.info("=" * 70)

    a_vals = [0.20, 0.30]; K = 100
    fig, axes = plt.subplots(len(a_vals), 3,
                              figsize=(15, 4 * len(a_vals)))

    for row, a in enumerate(a_vals):
        H = _a_to_H(a)
        D = a   # D = a = 1-H

        # ── Reference LP eigenvalues (density_simulation.py) ──────────
        lam_lp = eigenvalues_LP(a, K)
        var_lp = 2.0 * np.sum(lam_lp ** 2)
        lam_lp_n = lam_lp / np.sqrt(var_lp)   # normalised

        # ── Original cold-diffusion eigenvalues ────────────────────────
        from scipy.special import gamma as gamma_sc
        sigma_D   = np.sqrt((1.0 - 2.0 * D) * (1.0 - D) / 2.0)
        C_D_orig  = (2.0 * sigma_D
                     / (np.pi ** (1.0 - D) * gamma_sc(1.0 - D)
                        * np.sin(np.pi * D / 2.0)))
        n_arr     = np.arange(1, K + 1, dtype=float)
        corr      = np.sqrt(max(gamma_sc(D + 0.5) - 1.0, 0.0))
        lam_orig  = np.zeros(K)
        lam_orig[0] = (1 + 0.1409 * D) * np.sqrt(
            np.pi ** D * gamma_sc(1.0 - D)) * np.sqrt(0.5 - D)
        lam_orig[1:] = (C_D_orig * n_arr[1:] ** (D - 1.0)
                        + 1.25 * D ** 1.05 * corr * n_arr[1:] ** (D - 2.2))
        lam_orig  = np.maximum(lam_orig, 0.0)
        var_orig  = 2.0 * np.sum(lam_orig ** 2)
        lam_orig_n = lam_orig / np.sqrt(var_orig)  # normalised

        # ── (a) Eigenvalue comparison ──────────────────────────────────
        ax0 = axes[row, 0]
        ax0.semilogy(n_arr, lam_lp_n, "b-",  lw=2, label="LP reference (fixed)")
        ax0.semilogy(n_arr, lam_orig_n, "r--", lw=2, label="Original cold-diffusion")
        ax0.set_xlabel("n"); ax0.set_ylabel(r"$\lambda_n$ (normalised)")
        ax0.set_title(f"Eigenvalues  a={a}  H={H}")
        ax0.legend(fontsize=9); ax0.grid(True, alpha=0.3)
        rel_diff = np.abs(lam_lp_n - lam_orig_n) / (lam_lp_n + 1e-15)
        log.info("  a=%.2f  Max rel |λ_LP - λ_orig| / λ_LP: %.4f  Mean: %.4f",
                 a, np.max(rel_diff), np.mean(rel_diff))

        # ── (b) Density comparison ─────────────────────────────────────
        def chf_from_lam(lam_vec):
            def phi(z):
                z   = np.asarray(z, dtype=complex)
                chf = np.ones_like(z)
                for ln in lam_vec:
                    chf *= np.exp(-0.5 * np.log(1.0 - 2j * ln * z) - 1j * ln * z)
                return chf
            return phi

        x_lp, d_lp   = _density_from_chf(chf_from_lam(lam_lp_n),
                                           x_min=-3, x_max=6, N=2**15)
        x_or, d_or   = _density_from_chf(chf_from_lam(lam_orig_n),
                                           x_min=-3, x_max=6, N=2**15)
        ax1 = axes[row, 1]
        ax1.plot(x_lp, d_lp, "b-",  lw=2, label="LP (reference)")
        ax1.plot(x_or, d_or, "r--", lw=2, label="Original (biased)")
        ax1.set_xlabel("x"); ax1.set_ylabel("Density")
        ax1.set_title(f"Density comparison  a={a}")
        ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
        d_or_i   = np.interp(x_lp, x_or, d_or)
        max_dens = np.max(np.abs(d_lp - d_or_i))
        log.info("  a=%.2f  Max |density_LP - density_orig|: %.6f", a, max_dens)

        # ── (c) Cumulant differences ───────────────────────────────────
        ax2 = axes[row, 2]
        ks_range = np.arange(2, 8)
        kappa_lp  = np.array([2**(k-1)*math.factorial(k-1)*np.sum(lam_lp_n**k)
                               for k in ks_range])
        kappa_or  = np.array([2**(k-1)*math.factorial(k-1)*np.sum(lam_orig_n**k)
                               for k in ks_range])
        kappa_rel = np.abs(kappa_lp - kappa_or) / (np.abs(kappa_lp) + 1e-15)
        ax2.bar(ks_range - 0.2, kappa_lp, width=0.4, label="LP",       color='blue',  alpha=0.7)
        ax2.bar(ks_range + 0.2, kappa_or, width=0.4, label="Original",  color='red',   alpha=0.7)
        ax2.set_xlabel("Cumulant order k"); ax2.set_ylabel(r"$\kappa_k$")
        ax2.set_title(f"Cumulants  a={a}  (max rel diff={kappa_rel.max():.3f})")
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')
        log.info("  a=%.2f  Cumulant rel diffs: %s",
                 a, ", ".join([f"κ_{k}={v:.3f}" for k, v in zip(ks_range, kappa_rel)]))

    plt.suptitle(
        "FIX 4: Eigenvalue normalisation bias — original cold-diffusion vs LP reference\n"
        "Density and cumulant differences quantify the training distribution shift",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("../output/density/density_normalisation_discrepancy.png",
                dpi=150, bbox_inches="tight")
    log.info("Saved density_normalisation_discrepancy.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)

    log.info("Rosenblatt Density Simulation — Starting all experiments")
    log.info("=" * 70)

    t_total = time.time()

    experiment_fft_vs_quad()
    experiment_eigenvalue_comparison()
    experiment_compare_algorithms()
    experiment_density_multiple_H()
    experiment_cumulants()
    experiment_speed_benchmark()
    experiment_vt_convolution_vs_direct()
    experiment_validate_mc()
    experiment_exponential_bounds()
    experiment_density_discrepancy() 

    log.info("=" * 70)
    log.info("All experiments completed in %.1f s", time.time() - t_total)
    log.info("Output figures saved to ../output/density")
