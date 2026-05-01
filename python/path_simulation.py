"""
Rosenblatt Process — Path Simulation
======================================

Three algorithms for simulating paths of the Rosenblatt process Z_H(t):
 
  Algorithm 1 — Wavelet-Based Synthesis (Abry & Pipiras, 2000):
      Approximation II using FARIMA(0,κ,0) sequences and fractional
      wavelet filters. Complexity O(T · 2^J).
      FIX (vs original): centering constant `expected_sq` was the
      theoretical E[ξ²] of the raw FARIMA sequence, but after J
      wavelet upsampling steps E[ξ²] changes. We now estimate it
      empirically from 30 upsampled realisations during __init__.
 
  Algorithm 2 — Donsker-Type Approximation (Torres & Tudor, 2007):
      Discretization of the double Wiener–Itô integral via a random-walk
      approximation. Complexity O(N^3) for the full path.
      Status: correct as written; keep N small (≤ 60) for feasibility.
 
  Algorithm 3 — Markovian / Sum-of-Squares OU (Harms, 2019, corrected):
      Write Z_t = c(H,2) ∫₀ᵗ :V_u²: du where V_u is the causal
      Volterra factor with effective Hurst parameter H̃ = H − ½.
      Approximate its kernel K_{H̃}(r) = r^{H-1}/Γ(H) by n OU processes
      on Harms' geometric grid. Complexity O(n · M).
      FOUR BUGS fixed vs original (see class docstring).
 
Validation experiments check:
  • Marginal variance  Var Z(t) = t^{2H}
  • Cross-covariance   Cov(Z_s, Z_t) = ½(s^{2H}+t^{2H}−|s−t|^{2H})
  • Non-Gaussianity    skewness > 0, kurtosis > 3
  • Self-similarity     Z(at) =^d a^H Z(t)
  • Marginal density   compared to exact Veillette–Taqqu / LP formula
 
References
----------
  Abry & Pipiras (2000), Wavelet-based synthesis of the Rosenblatt process
  Torres & Tudor (2007), Donsker type theorem for the Rosenblatt process
  Harms (2019), Strong convergence rates for Markovian representations of fBm
  Veillette & Taqqu (2013), Properties and numerical evaluation of the
      Rosenblatt distribution
"""

import os
import time
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_fn
from scipy.stats import gaussian_kde, norm
from numba import njit

# ── logging ──────────────────────────────────────────────────
os.makedirs("../output/path/", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../output/path/path_simulation.log", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


###############################################################
#  Shared helpers
###############################################################

def rosenblatt_covariance(s, t, H):
    """R(s,t) = ½(|s|^{2H} + |t|^{2H} − |s−t|^{2H})."""
    H2 = 2.0 * H
    return 0.5 * (np.abs(s)**H2 + np.abs(t)**H2 - np.abs(s - t)**H2)


###############################################################
#  Algorithm 1 — Wavelet-Based Synthesis (Approximation II)
###############################################################

class WaveletRosenblatt:
    """
    Simulate Rosenblatt process paths via wavelet-based Approximation II.

    The Rosenblatt process is constructed as:
        Z_{κ,2}(t) = C_κ · 2^{−2κJ} · Σ_{0<i≤⌊2^J t⌋}
                      [ (ξ̃_i^(κ))² − E[ξ̃²] ]
 
    where ξ̃^(κ) is a FARIMA(0,κ,0) sequence refined through
    fractional wavelet filters from scale L to scale J.
 
    BUG FIX (vs original):
        The original code subtracted `expected_sq = Γ(1−2κ)/Γ(1−κ)²`,
        which is E[ξ²] for the *raw* FARIMA(0,κ,0) sequence. After J
        wavelet upsampling steps the marginal variance of ξ̃ changes, so
        subtracting the theoretical value introduces a systematic non-zero
        mean bias in Z(t). The fix is to estimate E[ξ̃²] empirically from
        30 upsampled realisations during __init__ and use that as the
        centering constant.

    Parameters
    ----------
    H : float
        Hurst parameter in (0.5, 1).
    N_vanishing : int
        Number of vanishing moments for Daubechies wavelet (default 2).
    J : int
        Final resolution scale. J=8 gives ~0.1 s/path; J=10 ~0.4 s/path.
    L : int
        Initial resolution scale (default 0).
    delta : float
        Truncation threshold for fractional filters.
    n_cal : int
        Number of paths used to calibrate C_κ (default 30).
    n_esq : int
        Number of realisations used to estimate E[ξ̃²] (default 30).
    """
 
    def __init__(self, H=0.7, N_vanishing=2, J=10, L=0, delta=1e-6,
                 n_cal=30, n_esq=30):
        assert 0.5 < H < 1.0, "H must be in (0.5, 1)"
        self.H = H
        self.kappa = H / 2.0           # κ = H/2 ∈ (1/4, 1/2)
        self.N_vanishing = N_vanishing
        self.J = J
        self.L = L
        self.delta = delta

        # Normalisation constant (will be calibrated empirically)
        k = self.kappa
        # Analytical formula: C_κ = Γ(κ) Γ(1−κ) √((4κ−1)κ) / Γ(1−2κ)
        self._C_kappa_analytical = (gamma_fn(k) * gamma_fn(1.0 - k)
                                    * np.sqrt((4.0 * k - 1.0) * k)
                                    / gamma_fn(1.0 - 2.0 * k))

        # Variance of each FARIMA term: E[ξ^(κ)²] = Γ(1−2κ)/Γ(1−κ)²
        self.expected_sq = gamma_fn(1.0 - 2.0 * k) / gamma_fn(1.0 - k)**2

        # Pre-compute Daubechies filters
        self._h, self._g = self._daubechies_filters(N_vanishing)

        # Pre-compute fractional filters u^(κ) and v^(κ)
        self._u_kappa, self._v_kappa = self._fractional_filters()

        # Empirical calibration: run a few test paths and adjust C_κ
        self.C_kappa = self._calibrate_normalization()

        # FIX: estimate E[ξ̃²] AFTER J wavelet upsampling steps
        # (not the theoretical FARIMA value, which changes under filtering)
        self.expected_sq = self._estimate_expected_sq(n_esq)
 
        # Empirical calibration: run a few test paths and adjust C_κ
        self.C_kappa = self._calibrate_normalization(n_cal)
 
 # ── Estimate E[ξ̃²] after upsampling ────────────────────
    def _estimate_expected_sq(self, n_est=30):
        """
        Estimate E[ξ̃²] for the wavelet-upsampled sequence ξ̃.
 
        After J upsampling steps the distribution of ξ changes and
        E[ξ̃²] ≠ Γ(1−2κ)/Γ(1−κ)² (the raw FARIMA value).
        We pool xi² over n_est independent realisations.
        """
        J = self.J; L = self.L; total_len = 2**J
        r = max(len(self._u_kappa), len(self._v_kappa))
        sq_vals = []
        for _ in range(n_est):
            xi = self._generate_farima(max(2**L + r, 64))
            for _ in range(J - L):
                eps = np.random.randn(len(xi))
                xi  = self._upsample_step(xi, eps)
            if len(xi) > total_len:   xi = xi[-total_len:]
            elif len(xi) < total_len: xi = np.concatenate(
                [np.zeros(total_len - len(xi)), xi])
            sq_vals.append(float(np.mean(xi**2)))
        return float(np.mean(sq_vals))
 
    # ── Calibration ──────────────────────────────────────────
    def _calibrate_normalization(self, n_cal=30):
        """
        Calibrate C_κ so that Var(Z(1)) ≈ 1.
        Uses the fixed expected_sq computed in __init__.
        """
        # Temporarily use analytical constant
        self.C_kappa = self._C_kappa_analytical

        samples = []
        for _ in range(n_cal):
            _, Z = self._simulate_path_internal(T=1.0)
            samples.append(Z[-1])

        samples = np.array(samples)
        emp_var = np.var(samples)
        target_var = 1.0  # Var(Z(1)) = 1^{2H} = 1

        if emp_var > 1e-12:
            correction = np.sqrt(1.0 / emp_var)
        else:
            correction = 1.0

        return self._C_kappa_analytical * correction

    # ── Daubechies filters ───────────────────────────────────
    @staticmethod
    def _daubechies_filters(N):
        """
        Return (h, g) low-pass and high-pass Daubechies filter coefficients
        with N vanishing moments. Uses pywt if available, otherwise
        hard-codes db1 / db2.

        For the wavelet SYNTHESIS (reconstruction) we use rec_lo / rec_hi,
        normalised so that Σ h_k² = 1 (energy normalisation).
        """
        try:
            import pywt
            w = pywt.Wavelet(f"db{N}")
            # rec_lo / rec_hi are the synthesis (reconstruction) filters
            # pywt filters satisfy Σ h² = 2; divide by √2 for unit energy
            h = np.array(w.rec_lo) / np.sqrt(2)
            g = np.array(w.rec_hi) / np.sqrt(2)
            return h, g
        except ImportError:
            pass

        # Fallback: hard-coded coefficients (normalised so Σh² = 1)
        if N == 1:   # Haar
            h = np.array([1.0, 1.0]) / np.sqrt(2)
        elif N == 2:  # db2
            h = np.array([
                0.6830127, 1.1830127, 0.3169873, -0.1830127
            ]) / np.sqrt(2)
        else:
            raise ValueError("Without pywt only N=1,2 supported")
        # QMF: g_k = (−1)^k h_{L-1-k}
        L = len(h)
        g = np.array([(-1)**k * h[L - 1 - k] for k in range(L)])
        return h, g

    # ── Fractional filter coefficients ───────────────────────
    def _fractional_filters(self):
        """
        Compute fractional wavelet filters u^(κ) and v^(κ).

        Following Abry & Pipiras (2000):
          u^(κ)(z) = (1 + z⁻¹)^κ · h(z)    [fractional low-pass]
          v^(κ)(z) = (1 − z⁻¹)^{−κ} · g(z)  [fractional high-pass]

        In the time domain these are convolutions of the base filters
        with the fractional coefficient sequences, truncated at δ.
        """
        k = self.kappa
        max_len = 512

        # ── (1 + z⁻¹)^κ coefficients ──
        # Generalised binomial: c_n = C(κ, n) = κ(κ-1)...(κ-n+1)/n!
        frac_u = np.zeros(max_len)
        frac_u[0] = 1.0
        for n in range(1, max_len):
            frac_u[n] = frac_u[n - 1] * (k - n + 1) / n

        # Truncate where |coeff| < delta
        idx = np.where(np.abs(frac_u) > self.delta)[0]
        frac_u = frac_u[:idx[-1] + 1] if len(idx) > 0 else frac_u[:1]

        # ── (1 − z⁻¹)^{−κ} coefficients ──
        # c_n = Γ(n+κ)/(Γ(κ)·n!) with recurrence: c_0=1, c_n = c_{n-1}·(n-1+κ)/n
        frac_v = np.zeros(max_len); frac_v[0] = 1.0

        for n in range(1, max_len):
            frac_v[n] = frac_v[n - 1] * (n - 1.0 + k) / n

        idx = np.where(np.abs(frac_v) > self.delta)[0]
        frac_v = frac_v[:idx[-1] + 1] if len(idx) > 0 else frac_v[:1]

        # Convolve with base wavelet filters
        u_kappa = np.convolve(frac_u, self._h)
        v_kappa = np.convolve(frac_v, self._g)

        return u_kappa, v_kappa

    # ── FARIMA(0,κ,0) via circulant embedding (FFT) ─────────
    def _generate_farima(self, length):
        """
        Generate a FARIMA(0,κ,0) stationary Gaussian sequence of given length
        using circulant embedding.

        The autocovariance is:
            γ(0) = Γ(1−2d)/Γ(1−d)²
            γ(k) = γ(k-1) * (k − 1 + d) / (k − d)   for k ≥ 1
        """
        d = self.kappa  # fractional differencing parameter
        n = length

        # Autocovariance function γ(k) for FARIMA(0,d,0)
        gamma_acf = np.zeros(n)
        gamma_acf[0] = gamma_fn(1.0 - 2.0 * d) / gamma_fn(1.0 - d)**2
        for kk in range(1, n):
            gamma_acf[kk] = gamma_acf[kk - 1] * (kk - 1.0 + d) / (kk - d)

        # Circulant embedding: embed in 2n circulant
        m = 2 * n
        row = np.zeros(m)
        row[:n] = gamma_acf
        row[n + 1:] = gamma_acf[1:][::-1]

        eigs = np.fft.fft(row).real
        if np.any(eigs < -1e-10):
            log.warning("FARIMA circulant eigs negative (min=%.2e), padding",
                        np.min(eigs))
            m2 = 4 * n
            row2 = np.zeros(m2)
            row2[:n] = gamma_acf
            row2[m2 - n + 1:] = gamma_acf[1:][::-1]
            eigs = np.fft.fft(row2).real
            m = m2
        eigs = np.maximum(eigs, 0.0)

        z = np.random.randn(m) + 1j * np.random.randn(m)
        w = np.fft.ifft(np.sqrt(eigs) * z)
        return w[:n].real

    # ── Wavelet upsampling recursion ─────────────────────────
    def _upsample_step(self, xi_prev, eps_prev):
        """
        One step of the fractional wavelet synthesis recursion:
            ξ_{j,·}^(κ) = u^(κ) * (↑2 ξ_{j-1,·}^(κ)) + v^(κ) * (↑2 ε_{j-1,·})
        """
        n_prev = len(xi_prev)
        n_next = 2 * n_prev

        # Upsample by 2 (insert zeros)
        up_xi = np.zeros(n_next)
        up_xi[::2] = xi_prev
        up_eps = np.zeros(n_next)
        up_eps[::2] = eps_prev

        # Convolve with fractional filters (use FFT for speed)
        fu = np.fft.fft(self._u_kappa, n=n_next)
        fv = np.fft.fft(self._v_kappa, n=n_next)
        fxi = np.fft.fft(up_xi)
        feps = np.fft.fft(up_eps)

        xi_next = np.fft.ifft(fu * fxi + fv * feps).real
        return xi_next

    # ── Full path simulation ─────────────────────────────────
    def _simulate_path_internal(self, T=1.0):
        """Internal path generation without time-scaling (for calibration)."""
        J = self.J
        L = self.L
        total_len = 2**J

        r = max(len(self._u_kappa), len(self._v_kappa))
        xi = self._generate_farima(max(2**L + r, 64))

        for _ in range(J-L):
            eps = np.random.randn(len(xi))
            xi = self._upsample_step(xi, eps)

        if len(xi) > total_len: xi = xi[-total_len:]
        elif len(xi) < total_len: xi = np.concatenate([np.zeros(total_len - len(xi)), xi])

        squared_centered = xi**2 - self.expected_sq
        S = np.cumsum(squared_centered)
        S = np.concatenate([[0.0], S])

        scale     = self.C_kappa * 2.0**(-2.0 * self.kappa * J)
        Z_raw     = scale * S * T**self.H
        raw_times = np.arange(total_len + 1) / float(2**J) * T

        return raw_times, Z_raw

    def simulate_path(self, T=1.0, n_points=None):
        """
        Simulate a path of the Rosenblatt process on [0, T].

        Parameters
        ----------
        T : float
            Time horizon.
        n_points : int or None
            Number of output time points (default 2^J + 1).

        Returns
        -------
        times : ndarray
        Z : ndarray
        """
        raw_times, Z_raw = self._simulate_path_internal(T)

        # Resample to n_points if requested
        if n_points is not None and n_points != len(raw_times):
            target_times = np.linspace(0, T, n_points)
            return target_times, np.interp(target_times, raw_times, Z_raw)

        return raw_times, Z_raw

    def simulate_paths_batch(self, T=1.0, n_points=200, n_paths=100):
        """Simulate multiple paths. Returns (times, paths_array)."""
        all_paths = []
        for _ in range(n_paths):
            t, Z = self.simulate_path(T=T, n_points=n_points)
            all_paths.append(Z)
        return t, np.array(all_paths)


###############################################################
#  Algorithm 2 — Donsker-Type Approximation
###############################################################

class DonskerRosenblatt:
    """
    Simulate Rosenblatt process paths via Donsker-type approximation.

    Z_t^n = Σ_{i≠j, 1≤i,j≤⌊nt⌋}  K_{i,j}^n(t) · ξ_i/√n · ξ_j/√n
    
    The kernel weights K_{i,j}^n(t) are:
        K_{i,j}^n(t) = n² ∫_{Δ_i} ∫_{Δ_j} Φ_t(y₁,y₂) dy₁ dy₂

    The full sum becomes:
        Z_t^n = (1/n) Σ_{i≠j} K_{i,j}^n(t) ξ_i ξ_j

    Parameters
    ----------
    H : float
        Hurst parameter in (0.5, 1).
    n_quad : int
        Number of quadrature nodes per dimension.
    """

    def __init__(self, H=0.7, n_quad=8):
        assert 0.5 < H < 1.0
        self.H = H
        self.Hp = (H + 1.0) / 2.0        # H' = (H+1)/2
        self.n_quad = n_quad

        # Quadrature nodes/weights
        self.nodes, self.weights = np.polynomial.legendre.leggauss(n_quad)

        # Cache for normalization constants at different N values
        self._c_H2_cache = {}

    # ── Normalization ────────────────────────────────────────
    def _get_normalization(self, N):
        """
        Get c(H,2) for discretisation level N, caching the result.

        For discretisation level N with ALL N cells in [0,1]:
            Z_1 = c · (1/N) Σ_{i≠j, i,j≤N} K_{ij}(1) ξ_i ξ_j

        By Isserlis' theorem:
            Var(Z_1) = c² · (2/N²) · Σ_{i≠j} K_{ij}²
        """
        if N in self._c_H2_cache:
            return self._c_H2_cache[N]

        dt_cell = 1.0 / N

        K = _build_donsker_kernel(N, 1.0, dt_cell, self.Hp,
                                  self.nodes, self.weights, self.n_quad)
        # Frobenius norm of off-diagonal part
        frob = sum(K[i, j]**2 for i in range(N) for j in range(N) if i != j)
        raw_var = 2.0 * frob / N**2
        c_H2 = 1.0 / np.sqrt(raw_var) if raw_var > 0 else 1.0

        self._c_H2_cache[N] = c_H2
        return c_H2

    # ── Path simulation ──────────────────────────────────────
    def simulate_path(self, T=1.0, N=50):
        """
        Simulate a path on [0, T] with N time steps.

        We build the FULL N×N kernel matrix once for t=T, then for each
        time t_k = k·T/N, we sum only over indices i,j ≤ k.

        The cells Δ_i = [(i-1)·T/N, i·T/N] are fixed throughout.
        K_{ij}(T) = N² ∫_{Δ_i} ∫_{Δ_j} Φ_T(y₁,y₂) dy₁ dy₂

        For intermediate times, we use K_{ij}(t_k) which we approximate
        by building a separate kernel with upper limit t_k.
        """
        xi = np.random.randn(N)
        dt_cell = T / N
        times = np.linspace(0, T, N + 1)
        Z = np.zeros(N + 1)

        # Get normalization for this N (calibrated at T=1, t=1)
        c_H2 = self._get_normalization(N)

        # Build kernel for each time step
        for k in range(1, N + 1):
            t_k = times[k]
            # Build full k×k kernel for cells [0, dt_cell], ..., [(k-1)*dt_cell, k*dt_cell]
            # with integration upper limit t_k
            K_k = _build_donsker_kernel(k, t_k, dt_cell, self.Hp,
                                        self.nodes, self.weights, self.n_quad)

            # K_k[i,j] = k² ∫∫, but we want N² ∫∫ for consistent normalization
            correction = float(N * N) / float(k * k) 
            
            val = sum(K_k[i, j] * xi[i] * xi[j]
                      for i in range(k) for j in range(k) if i != j)
            
            Z[k] = c_H2 * correction * val / N

        return times, Z

    def simulate_paths_batch(self, T=1.0, N=50, n_paths=100):
        """Simulate multiple Donsker paths."""
        all_Z = []
        for _ in range(n_paths):
            t, Z = self.simulate_path(T=T, N=N)
            all_Z.append(Z)
        return t, np.array(all_Z)


@njit
def _build_donsker_kernel(n, t, dt_cell, Hp, nodes, weights, nq):
    """
    Numba-accelerated Donsker kernel construction.

    Build n×n kernel matrix K where:
        K[i,j] = n² ∫_{Δ_i} ∫_{Δ_j} ∫_{y1∨y2}^{t} f^{H'}(u,y1)·f^{H'}(u,y2) du dy1 dy2

    Grid cells: Δ_i = [i·dt_cell, (i+1)·dt_cell], i = 0,...,n-1
    """
    K = np.zeros((n, n))
    alpha = Hp - 1.5

    for i in range(n):
        y1_lo = i * dt_cell
        y1_hi = (i + 1) * dt_cell
        y1_mid = 0.5 * (y1_hi + y1_lo)
        y1_half = 0.5 * (y1_hi - y1_lo)

        for j in range(n):
            if i == j:
                continue
            y2_lo = j * dt_cell
            y2_hi = (j + 1) * dt_cell
            y2_mid  = 0.5 * (y2_hi + y2_lo)
            y2_half = 0.5 * (y2_hi - y2_lo)

            # Quadrature over Δ_i × Δ_j × [y1∨y2, t]
            total = 0.0
            for qi in range(nq):
                y1 = y1_mid + y1_half * nodes[qi]
                if y1 <= 0:
                    continue
                w1 = weights[qi] * y1_half

                for qj in range(nq):
                    y2 = y2_mid + y2_half * nodes[qj]
                    if y2 <= 0:
                        continue
                    w2 = weights[qj] * y2_half

                    lower = max(y1, y2)
                    if lower >= t:
                        continue
                    u_mid = 0.5 * (t + lower)
                    u_half = 0.5 * (t - lower)

                    inner = 0.0
                    for qu in range(nq):
                        u = u_mid + u_half * nodes[qu]
                        # f^{H'}(u,y1) * f^{H'}(u,y2)
                        f1 = (u / y1)**(0.5 - Hp) * (u - y1)**alpha
                        f2 = (u / y2)**(0.5 - Hp) * (u - y2)**alpha
                        inner += weights[qu] * f1 * f2
                    inner *= u_half

                    total += w1 * w2 * inner

            K[i, j] = n * n * total
    return K


###############################################################
#  Algorithm 3 — Markovian / Sum-of-Squares OU  (CORRECTED)
###############################################################
 
class MarkovianRosenblatt:
    """
    Simulate the Rosenblatt process via Markovian sum-of-squares OU.
 
    Mathematical foundation
    -----------------------
    Z_t^{H,2} = c(H,2) ∫₀ᵗ :V_u²: du,
    where V_u = ∫₀ᵘ K_{H̃}(u−s) dW(s), K_{H̃}(r) = r^{H-1}/Γ(H),
    and  H̃ = H − ½ ∈ (0, ½)  is the effective Hurst parameter.
 
    Harms' geometric grid with γ = 1−H, δ = H−½ approximates K_{H̃}
    by n OU processes (all sharing one Brownian motion):
 
        Z_t^{H,n} = c(H,2) ∫₀ᵗ [(Σₖ wₖ U_s^(k))² − φₙ(s)] ds
 
    FOUR BUGS FIXED vs the original uploaded code
    -----------------------------------------------
    Bug 1 (critical): original used Hp = (H+1)/2 → exponent H/2−1 < −½,
        kernel NOT in L²([0,u]). Fix: H_tilde = H − ½.
 
    Bug 2: Harms grid used wrong exponents derived from Hp.
        Fix: γ = 1−H, δ = H−½, x_min = n^{−r/γ}, x_max = n^{r/δ}.
 
    Bug 3: OU noise coefficient (1−e^{−x_k Δt})/(x_k Δt) diverges
        for fast modes. Fix: U = U·decay + ΔW  (coefficient = 1),
        with cap x_max · Δt ≤ 0.1 to keep Euler self-consistent.
 
    Bug 4: calibration used 2∫φₙ(s)² ds (diagonal only).
        Fix: Var(Z_T) = c² · 2∫∫(E[V_s V_t])² ds dt (full isometry).
 
    Known limitation — path smoothness
    ------------------------------------
    Markovian paths look visually smoother than Wavelet paths.
    This is a THEORETICAL limitation, not a code bug: the kernel
    K_{H̃}(r) = r^{H−1}/Γ(H) → ∞ as r → 0 (singular), creating
    Hölder roughness. Any finite exponential sum K_n(0) < ∞ cannot
    reproduce this singularity. The marginal DISTRIBUTION at fixed t
    can still be approximately correct (Prof. Ivan's test), even though
    individual path realisations look smoother.
 
    Parameters
    ----------
    H : float           Hurst parameter in (0.5, 1).
    n_modes : int       Number of OU modes (default 50).
    r : float           Superconvergence exponent for Harms' grid (default 2).
    T : float           Time horizon used to set the grid cap x_max·Δt ≤ 0.1.
    n_times : int       Number of time steps (sets Δt = T/n_times).
    """
 
    def __init__(self, H=0.7, n_modes=50, r=2.0, T=1.0, n_times=500):
        assert 0.5 < H < 1.0
        self.H       = H
        self.H_tilde = H - 0.5          # H̃ = H − ½ ∈ (0, ½)
        self.n_modes = n_modes
        self.r       = r
        self._T0     = T
        self._M0     = n_times
        self._build_grid(T, n_times)
 
    # ── Grid construction ────────────────────────────────────
    def _build_grid(self, T, n_times):
        """
        Harms geometric grid for K_{H̃}(r) = r^{H̃−½}/Γ(H̃+½) = r^{H−1}/Γ(H).
 
        Laplace measure: μ_{H̃}(dx) = x^{−(H̃+½)} / (Γ(H̃+½)·Γ(½−H̃)) dx
 
        Grid parameters (Harms 2019, Theorem 1):
            γ = ½ − H̃ = 1 − H  →  x_min = n^{−r/γ}
            δ = H̃ = H − ½      →  x_max = n^{r/δ}
 
        Cap: x_max · Δt ≤ 0.1  (Euler / continuous covariance consistent).
 
        Weights (midpoint quadrature on log scale):
            w_k = x_k^{½−H̃} · Δlog / (Γ(H̃+½)·Γ(½−H̃))
        """
        Ht  = self.H_tilde
        n   = self.n_modes
        r   = self.r
        dt  = T / n_times
 
        gamma_par = 0.5 - Ht    # = 1 − H  ∈ (0, ½)
        delta_par = Ht          # = H − ½  ∈ (0, ½)
 
        x_min = max(n ** (-r / gamma_par), 1e-10)
        x_max = min(n ** ( r / delta_par), 0.1 / dt)   # Euler cap
        if x_max <= x_min:
            x_min, x_max = 1e-4, max(10.0 / T, 1e-3)
 
        log_x = np.linspace(np.log(x_min), np.log(x_max), n)
        x_k   = np.exp(log_x)
        dlog  = (log_x[-1] - log_x[0]) / (n - 1) if n > 1 else 1.0
 
        norm_c = gamma_fn(Ht + 0.5) * gamma_fn(0.5 - Ht)
        w_k    = x_k ** (0.5 - Ht) * dlog / norm_c
 
        self.speeds      = x_k
        self.weights_ou  = w_k
        self._w_outer    = np.outer(w_k, w_k)          # (n, n)
        self._sum_x      = x_k[:, None] + x_k[None, :] # (n, n)
 
    # ── Covariance of V^(n) ──────────────────────────────────
    def _cov_Vs_Vt(self, s, t):
        """E[V_s^(n) · V_t^(n)] for s ≤ t via continuous OU formula."""
        if s > t: s, t = t, s
        base  = (1.0 - np.exp(-self._sum_x * s)) / self._sum_x
        decay = np.exp(-self.speeds * (t - s))
        return float(np.sum(self._w_outer * base * decay[None, :]))
 
    def _precompute_variance(self, times):
        """
        φₙ(t) = E[Ṽ(t)²] = Σ_{k,l} w_k w_l (1−e^{−(x_k+x_l)t})/(x_k+x_l)
 
        Consistent with the Euler scheme because x_max · Δt ≤ 0.1.
        """
        va = np.zeros(len(times))
        for i, t in enumerate(times):
            if t > 0:
                cov_mat  = (1.0 - np.exp(-self._sum_x * t)) / self._sum_x
                va[i]    = float(np.sum(self._w_outer * cov_mat))
        return va
 
    # ── Calibration ──────────────────────────────────────────
    def _calibrate_normalization(self, T):
        """
        Compute c(H,2) so that Var(Z_T^{H,n}) = T^{2H}.
 
        Second Wiener-chaos isometry:
            Var(Z_T) = c² · 2 ∫₀ᵀ ∫₀ᵀ (E[V_s V_t])² ds dt
        evaluated on an 80-point Riemann grid.
        """
        nc     = 80
        dt_cal = T / nc
        t_cal  = np.arange(1, nc + 1) * dt_cal
 
        EVsVt = np.zeros((nc, nc))
        for i in range(nc):
            base_i        = (1.0 - np.exp(-self._sum_x * t_cal[i])) / self._sum_x
            EVsVt[i, i]   = float(np.sum(self._w_outer * base_i))
            for j in range(i + 1, nc):
                v             = self._cov_Vs_Vt(t_cal[i], t_cal[j])
                EVsVt[i, j]   = v
                EVsVt[j, i]   = v
 
        integral   = 2.0 * np.sum(EVsVt ** 2) * dt_cal ** 2
        target_var = T ** (2.0 * self.H)
        return np.sqrt(target_var / integral) if integral > 1e-15 else 1.0
 
    # ── Path simulation ──────────────────────────────────────
    def simulate_path(self, T=1.0, n_times=500):
        """
        Simulate a single path of the Rosenblatt process on [0, T].
 
        OU update (exact-mean Euler, shared scalar increment):
            ΔW_j ~ N(0, Δt),  same for ALL modes k
            U^(k)_{j+1} = e^{−x_k Δt} · U^(k)_j + ΔW_j
        """
        if T != self._T0 or n_times != self._M0:
            self._T0 = T; self._M0 = n_times
            self._build_grid(T, n_times)
 
        dt    = T / n_times
        times = np.linspace(0, T, n_times + 1)
        va    = self._precompute_variance(times)
        c_H2  = self._calibrate_normalization(T)
        decay = np.exp(-self.speeds * dt)       # (n,)
 
        U = np.zeros(self.n_modes)
        Z = np.zeros(n_times + 1)
 
        for j in range(n_times):
            dW       = np.random.randn() * np.sqrt(dt)  # shared increment
            U        = U * decay + dW                    # vectorised OU step
            V        = np.dot(self.weights_ou, U)
            Z[j + 1] = Z[j] + c_H2 * (V ** 2 - va[j + 1]) * dt
 
        return times, Z
 
    def simulate_paths_batch(self, T=1.0, n_times=500, n_paths=100):
        """
        Simulate n_paths independent paths.
        Grid, variance array, and c(H,2) are computed once and reused.
        Complexity O(n_modes · n_times · n_paths).
        """
        if T != self._T0 or n_times != self._M0:
            self._T0 = T; self._M0 = n_times
            self._build_grid(T, n_times)
 
        dt    = T / n_times
        times = np.linspace(0, T, n_times + 1)
        va    = self._precompute_variance(times)
        c_H2  = self._calibrate_normalization(T)
        decay = np.exp(-self.speeds * dt)       # (n,)
 
        Z_all = np.zeros((n_paths, n_times + 1))
        for p in range(n_paths):
            U = np.zeros(self.n_modes)
            for j in range(n_times):
                dW             = np.random.randn() * np.sqrt(dt)
                U              = U * decay + dW
                V              = np.dot(self.weights_ou, U)
                Z_all[p, j+1]  = Z_all[p, j] + c_H2 * (V**2 - va[j+1]) * dt
 
        return times, Z_all


###############################################################
#  Experiments
###############################################################

def experiment_sample_paths():
    """Experiment 1: Plot sample paths from all three methods."""
    log.info("=" * 70)
    log.info("Experiment 1: Sample paths from all three algorithms")
    log.info("=" * 70)

    H = 0.7
    n_paths = 5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Wavelet ---
    log.info("Wavelet paths ...")
    t0 = time.time()
    wav = WaveletRosenblatt(H=H, J=10, L=0, N_vanishing=2)
    for i in range(n_paths):
        t_w, Z_w = wav.simulate_path(T=1.0, n_points=200)
        axes[0].plot(t_w, Z_w, alpha=0.7, linewidth=1.0)
    dt_w = time.time() - t0
    axes[0].set_title("Wavelet (J=10)", fontsize=13)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("$Z_t$")
    axes[0].grid(True, alpha=0.3)
    log.info("  Wavelet: %.2f s for %d paths", dt_w, n_paths)

    # --- Donsker ---
    log.info("Donsker paths ...")
    t0 = time.time()
    don = DonskerRosenblatt(H=H, n_quad=6)
    for i in range(n_paths):
        t_d, Z_d = don.simulate_path(T=1.0, N=60)
        axes[1].plot(t_d, Z_d, alpha=0.7, linewidth=1.0)
    dt_d = time.time() - t0
    axes[1].set_title("Donsker (N=60)", fontsize=13)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("$Z_t$")
    axes[1].grid(True, alpha=0.3)
    log.info("  Donsker: %.2f s for %d paths", dt_d, n_paths)

    # --- Markovian OU ---
    log.info("Markovian OU paths ...")
    t0 = time.time()
    mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=500)
    for i in range(n_paths):
        t_m, Z_m = mou.simulate_path(T=1.0, n_times=500)
        axes[2].plot(t_m, Z_m, alpha=0.7, linewidth=1.0)
    dt_m = time.time() - t0
    axes[2].set_title("Markovian OU (n=50, experimental)", fontsize=13)
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("$Z_t$")
    axes[2].grid(True, alpha=0.3)
    log.info("  Markovian OU: %.2f s for %d paths", dt_m, n_paths)

    plt.suptitle(f"Rosenblatt Process Sample Paths (H={H})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../output/path/path_sample_paths.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_sample_paths.png")
    plt.close()


def experiment_variance():
    """Experiment 2: Check Var(Z(t)) = t^{2H}."""
    log.info("=" * 70)
    log.info("Experiment 2: Marginal variance Var(Z(t)) = t^{2H}")
    log.info("=" * 70)

    H = 0.7
    n_paths = 500
    t_test = [0.25, 0.5, 0.75, 1.0]

    methods = {}

    # --- Wavelet ---
    log.info("Wavelet: generating %d paths ...", n_paths)
    t0 = time.time()
    wav = WaveletRosenblatt(H=H, J=10, L=0)
    t_w_grid, wav_paths = wav.simulate_paths_batch(T=1.0, n_points=200,
                                                   n_paths=n_paths)
    methods["Wavelet"] = (t_w_grid, wav_paths)
    log.info("  Wavelet done in %.1f s", time.time() - t0)

    # --- Donsker ---
    n_don = min(n_paths, 200)
    log.info("Donsker: generating %d paths ...", n_don)
    t0 = time.time()
    don = DonskerRosenblatt(H=H, n_quad=6)
    t_d_grid, don_paths = don.simulate_paths_batch(T=1.0, N=60, n_paths=n_don)
    methods["Donsker"] = (t_d_grid, don_paths)
    log.info("  Donsker done in %.1f s", time.time() - t0)

    # --- Markovian OU ---
    log.info("Markovian OU: generating %d paths ...", n_paths)
    t0 = time.time()
    mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=500)
    t_m, Z_all_m = mou.simulate_paths_batch(T=1.0, n_times=500, n_paths=n_paths)
    methods["Markovian OU"] = (t_m, Z_all_m)
    log.info("  Markovian OU done in %.1f s", time.time() - t0)

    # Report
    log.info("")
    header = f"{'Method':<16s} {'t':>5s} {'Var_emp':>10s} {'Var_th':>10s} {'Ratio':>8s}"
    log.info(header)
    log.info("-" * len(header))

    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    t_theory = np.linspace(0.01, 1.0, 100)
    ax.plot(t_theory, t_theory**(2 * H), "k-", linewidth=2, label="Theory $t^{2H}$")

    colors = {"Wavelet": "blue", "Donsker": "red", "Markovian OU": "green"}

    for name, (t_grid, paths) in methods.items():
        var_curve = np.var(paths, axis=0)
        for tt in t_test:
            idx   = np.argmin(np.abs(t_grid - tt))
            v_emp = var_curve[idx]
            v_th = tt**(2 * H)
            log.info(f"  {name:<16s} {tt:5.2f} {v_emp:10.4f} {v_th:10.4f} {v_emp / v_th:8.3f}")

        ax.plot(t_grid, var_curve, "--", linewidth=1.5, color=colors[name],
                label=name)

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("Var($Z_t$)", fontsize=12)
    ax.set_title(f"Marginal Variance (H={H})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/path/path_variance.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_variance.png")
    plt.close()

    return methods


def experiment_covariance(methods=None):
    """Experiment 3: Check cross-covariance."""
    log.info("=" * 70)
    log.info("Experiment 3: Cross-covariance check")
    log.info("=" * 70)

    H = 0.7

    if methods is None:
        n_paths = 300
        wav = WaveletRosenblatt(H=H, J=10, L=0)
        t_w, wav_paths = wav.simulate_paths_batch(T=1.0, n_points=200, n_paths=n_paths)
        methods = {"Wavelet": (t_w, wav_paths)}

        mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=500)
        t_m, Z_m = mou.simulate_paths_batch(T=1.0, n_times=500, n_paths=n_paths)
        methods["Markovian OU"] = (t_m, Z_m)

    pairs = [(0.25, 0.5), (0.25, 0.75), (0.5, 1.0), (0.25, 1.0)]

    log.info("")
    header = f"{'Method':<16s} {'s':>5s} {'t':>5s} {'Cov_emp':>10s} {'Cov_th':>10s} {'Ratio':>8s}"
    log.info(header)
    log.info("-" * len(header))

    for name, (t_grid, paths) in methods.items():
        for s, t in pairs:
            si  = np.argmin(np.abs(t_grid - s))
            ti  = np.argmin(np.abs(t_grid - t))
            emp = np.mean(paths[:, si] * paths[:, ti])
            th  = rosenblatt_covariance(t_grid[si], t_grid[ti], H)
            ratio = emp / th if abs(th) > 1e-12 else float("nan")
            log.info(f"  {name:<16s} {s:5.2f} {t:5.2f} {emp:10.4f} {th:10.4f} {ratio:8.3f}")


def experiment_non_gaussianity():
    """Experiment 4: Skewness and kurtosis at t=1."""
    log.info("=" * 70)
    log.info("Experiment 4: Non-Gaussianity (skewness, kurtosis)")
    log.info("=" * 70)

    H = 0.7
    n_paths = 2000

    results = {}

    # Wavelet
    log.info("  Wavelet: %d samples ...", n_paths)
    wav = WaveletRosenblatt(H=H, J=10, L=0)
    _, wav_p = wav.simulate_paths_batch(T=1.0, n_points=2, n_paths=n_paths)
    results["Wavelet"] = wav_p[:, -1]

    # Donsker (fewer)
    n_don = min(n_paths, 500)
    log.info("  Donsker: %d samples ...", n_don)
    don = DonskerRosenblatt(H=H, n_quad=6)
    _, don_p = don.simulate_paths_batch(T=1.0, N=60, n_paths=n_don)
    results["Donsker"] = don_p[:, -1]

    # Markovian OU
    log.info("  Markovian OU: %d samples ...", n_paths)
    mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=500)
    _, Z_m = mou.simulate_paths_batch(T=1.0, n_times=500, n_paths=n_paths)
    results["Markovian OU"] = Z_m[:, -1]

    log.info("")
    header = f"{'Method':<16s} {'Mean':>8s} {'Std':>8s} {'Skew':>8s} {'Kurt':>8s}"
    log.info(header)
    log.info("-" * len(header))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Wavelet": "blue", "Donsker": "red", "Markovian OU": "green"}

    for idx, (name, samp) in enumerate(results.items()):
        mu = np.mean(samp)
        std = np.std(samp)
        if std > 0:
            skew = float(np.mean(((samp - mu) / std)**3))
            kurt = float(np.mean(((samp - mu) / std)**4))
        else:
            skew = kurt = float("nan")
        log.info(f"  {name:<16s} {mu:8.4f} {std:8.4f} {skew:8.3f} {kurt:8.3f}")

        ax = axes[idx]
        ax.hist(samp, bins=60, density=True, alpha=0.6, color=colors[name],
                label=name)
        x_g = np.linspace(mu - 4 * std, mu + 4 * std, 200)
        ax.plot(x_g, norm.pdf(x_g, mu, std), "k--", linewidth=1.5,
                label="Gaussian")
        ax.set_title(f"{name}\nskew={skew:.2f}, kurt={kurt:.2f}", fontsize=11)
        ax.set_xlabel("$Z_1$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    log.info("  Theory (a=0.3): skew ≈ 1.9, kurt ≈ 9.5")

    plt.suptitle(f"Marginal Distribution at t=1 (H={H})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../output/path/path_non_gaussianity.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_non_gaussianity.png")
    plt.close()

    return results


def experiment_self_similarity():
    """Experiment 5: Self-similarity Z(at) =^d a^H Z(t).

    Uses batch generation with REDUCED sample counts for speed.
    Donsker is O(N³) per path, so we use far fewer samples.
    """
    log.info("=" * 70)
    log.info("Experiment 5: Self-similarity")
    log.info("=" * 70)

    H = 0.7
    a_scale = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Wavelet (fast: ~0.4s per path) ----
    n_wav = 500  # Reduced from 2000
    log.info("  Wavelet: generating %d pairs ...", n_wav)
    wav = WaveletRosenblatt(H=H, J=10, L=0)
    t0 = time.time()

    # Z(a·T): simulate on [0, a], take endpoint
    _, paths_a = wav.simulate_paths_batch(T=a_scale, n_points=2, n_paths=n_wav)
    Z_a_wav = paths_a[:, -1]

    # a^H · Z(T): simulate on [0, 1], take endpoint, scale by a^H
    _, paths_1 = wav.simulate_paths_batch(T=1.0, n_points=2, n_paths=n_wav)
    Z_1_wav = a_scale**H * paths_1[:, -1]

    log.info("    Wavelet done in %.1f s", time.time() - t0)

    axes[0].hist(Z_a_wav, bins=40, density=True, alpha=0.5,
                 label=f"$Z({a_scale})$")
    axes[0].hist(Z_1_wav, bins=40, density=True, alpha=0.5,
                 label=f"${a_scale}^H Z(1)$")
    axes[0].set_title("Wavelet", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    log.info("    Z(%.1f) std=%.3f, a^H·Z(1) std=%.3f",
             a_scale, np.std(Z_a_wav), np.std(Z_1_wav))

    # ---- Donsker (slow: ~2s per path at N=60) ----
    n_don = 100  # Reduced from 2000 to ~200s total
    log.info("  Donsker: generating %d pairs ...", n_don)
    don = DonskerRosenblatt(H=H, n_quad=6)
    t0 = time.time()

    _, paths_a_d = don.simulate_paths_batch(T=a_scale, N=60, n_paths=n_don)
    Z_a_don = paths_a_d[:, -1]

    _, paths_1_d = don.simulate_paths_batch(T=1.0, N=60, n_paths=n_don)
    Z_1_don = a_scale**H * paths_1_d[:, -1]

    log.info("    Donsker done in %.1f s", time.time() - t0)

    axes[1].hist(Z_a_don, bins=30, density=True, alpha=0.5,
                 label=f"$Z({a_scale})$")
    axes[1].hist(Z_1_don, bins=30, density=True, alpha=0.5,
                 label=f"${a_scale}^H Z(1)$")
    axes[1].set_title("Donsker (N=60, fewer samples)", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    log.info("    Z(%.1f) std=%.3f, a^H·Z(1) std=%.3f",
             a_scale, np.std(Z_a_don), np.std(Z_1_don))

    plt.suptitle(
        f"Self-Similarity: $Z({a_scale})$ vs ${a_scale}^H Z(1)$ (H={H})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("../output/path/path_self_similarity.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_self_similarity.png")
    plt.close()


def experiment_density_comparison():
    """Experiment 6: Compare marginal density at t=1 with exact density."""
    log.info("=" * 70)
    log.info("Experiment 6: Density at t=1 vs exact (LP / VT)")
    log.info("=" * 70)

    H = 0.7
    a = 1.0 - H
    n_paths = 3000

    # Exact density from density_simulation.py
    try:
        from python.density_simulation import RosenblattDensityLP
        lp = RosenblattDensityLP(a=a, K=200)
        x_exact, d_exact = lp.density_fft(x_min=-3.0, x_max=6.0)
        have_exact = True
        log.info("  Loaded exact density from density_simulation.py")
    except ImportError:
        log.warning("  Cannot import density_simulation — using fallback")
        have_exact = False

    # Generate samples
    log.info("  Generating samples ...")
    wav = WaveletRosenblatt(H=H, J=10, L=0)
    don = DonskerRosenblatt(H=H, n_quad=6)
    mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=500)

    samples = {}

    t0 = time.time()
    _, wav_p = wav.simulate_paths_batch(T=1.0, n_points=2, n_paths=n_paths)
    samples["Wavelet"] = wav_p[:, -1]
    log.info("    Wavelet: %.1f s", time.time() - t0)

    n_don = min(n_paths, 800)
    t0 = time.time()
    _, don_p = don.simulate_paths_batch(T=1.0, N=60, n_paths=n_don)
    samples["Donsker"] = don_p[:, -1]
    log.info("    Donsker: %.1f s", time.time() - t0)

    t0 = time.time()
    _, Z_m = mou.simulate_paths_batch(T=1.0, n_times=500, n_paths=n_paths)
    samples["Markovian OU"] = Z_m[:, -1]
    log.info("    Markovian OU: %.1f s", time.time() - t0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Wavelet": "blue", "Donsker": "red", "Markovian OU": "green"}

    for idx, (name, samp) in enumerate(samples.items()):
        ax = axes[idx]
        ax.hist(samp, bins=80, density=True, alpha=0.4, color=colors[name],
                label=f"{name} histogram")
        if have_exact:
            ax.plot(x_exact, d_exact, "k-", linewidth=2, label="Exact (LP)")
        try:
            kde = gaussian_kde(samp, bw_method="scott")
            x_kde = np.linspace(-3, 6, 300)
            ax.plot(x_kde, kde(x_kde), "--", color=colors[name],
                    linewidth=1.5, label="KDE")
        except Exception:
            pass

        ax.set_title(name, fontsize=13)
        ax.set_xlabel("x")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Marginal Density at t=1 (H={H})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../output/path/path_density_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_density_comparison.png")
    plt.close()


def experiment_speed_comparison():
    """Experiment 7: Wall-clock speed comparison."""
    log.info("=" * 70)
    log.info("Experiment 7: Speed comparison")
    log.info("=" * 70)

    H = 0.7
    n_points_list = [50, 100, 200, 500]

    results = {name: [] for name in ["Wavelet", "Donsker", "Markovian OU"]}

    for npt in n_points_list:
        # Wavelet
        wav = WaveletRosenblatt(H=H, J=max(int(np.ceil(np.log2(npt))), 6),
                                L=0)
        t0 = time.time()
        for _ in range(5):
            wav.simulate_path(T=1.0, n_points=npt)
        dt_w = (time.time() - t0) / 5
        results["Wavelet"].append(dt_w)

        # Donsker (only for small N due to O(N³) cost)
        N_don = min(npt, 80)
        don = DonskerRosenblatt(H=H, n_quad=6)
        t0 = time.time()
        for _ in range(3):
            don.simulate_path(T=1.0, N=N_don)
        dt_d = (time.time() - t0) / 3
        results["Donsker"].append(dt_d)

        # Markovian OU
        mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=npt)
        t0 = time.time()
        for _ in range(5):
            mou.simulate_path(T=1.0, n_times=npt)
        dt_m = (time.time() - t0) / 5
        results["Markovian OU"].append(dt_m)

        log.info("  n=%d: Wav=%.3fs  Don=%.3fs  MOU=%.3fs",
                 npt, dt_w, dt_d, dt_m)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, times_list in results.items():
        ax.plot(n_points_list, times_list, "o-", linewidth=2, markersize=6,
                label=name)
    ax.set_xlabel("Number of output points", fontsize=12)
    ax.set_ylabel("Time per path (s)", fontsize=12)
    ax.set_title(f"Speed Comparison (H={H})", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/path/path_speed_comparison.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_speed_comparison.png")
    plt.close()


def experiment_method_comparison_overlay():
    """Experiment 8: Overlay single paths from all methods."""
    log.info("=" * 70)
    log.info("Experiment 8: Method comparison overlay")
    log.info("=" * 70)

    H = 0.7
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    wav = WaveletRosenblatt(H=H, J=12, L=0)
    t_w, Z_w = wav.simulate_path(T=1.0, n_points=500)
    ax.plot(t_w, Z_w, "b-", linewidth=1.2, alpha=0.8, label="Wavelet (J=12)")

    don = DonskerRosenblatt(H=H, n_quad=8)
    t_d, Z_d = don.simulate_path(T=1.0, N=60)
    ax.plot(t_d, Z_d, "r-", linewidth=1.2, alpha=0.8, label="Donsker (N=60)")

    mou = MarkovianRosenblatt(H=H, n_modes=50, T=1.0, n_times=500)
    t_m, Z_m = mou.simulate_path(T=1.0, n_times=500)
    ax.plot(t_m, Z_m, "g-", linewidth=1.2, alpha=0.8,
            label="Markovian OU (n=50, experimental)")

    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("$Z_t$", fontsize=12)
    ax.set_title(f"Rosenblatt Paths: Three Methods (H={H})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../output/path/path_method_overlay.png", dpi=150, bbox_inches="tight")
    log.info("Saved ../output/path/path_method_overlay.png")
    plt.close()


###############################################################
#  Main
###############################################################

if __name__ == "__main__":
    np.random.seed(42)

    log.info("Rosenblatt Path Simulation — Starting experiments")
    log.info("=" * 70)
    t_total = time.time()

    # Warm up Numba JIT
    log.info("Warming up Numba JIT ...")
    _don = DonskerRosenblatt(H=0.7, n_quad=4)
    _don.simulate_path(T=1.0, N=10)
    log.info("JIT warm-up done.\n")

    experiment_sample_paths()
    methods = experiment_variance()
    experiment_covariance(methods)
    experiment_non_gaussianity()
    experiment_self_similarity()
    experiment_density_comparison()
    experiment_speed_comparison()
    experiment_method_comparison_overlay()

    log.info("=" * 70)
    log.info("All experiments completed in %.1f s", time.time() - t_total)
    log.info("Output figures saved to ../output/path/")
    log.info("Log saved to ../output/path/path_simulation.log")