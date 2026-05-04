# Proton NMR perturbation — summary & next steps

**Primary reference (equation numbers below):** J. A. Pople, “Molecular-Orbital Theory of Diamagnetism. I. An Approximate LCAO Scheme,” *J. Chem. Phys.* **37**, 53 (1962) (DOI [10.1063/1.1732974](https://doi.org/10.1063/1.1732974)); local copy `pople1962.pdf`.

## To-do (next implementation steps)

- [ ] Fix **$B_k \leftrightarrow \lambda$** convention: Fock uses $i \lambda L_k$; finite differences must produce $\partial P / \partial B_k$ (include factor $d\lambda / dB_k$ if $\lambda$ is not literal $B_k$).
- [ ] Add **pure** finite-difference helpers (e.g. central $(P_+ - P_-)/(2\varepsilon)$) — ideally in `nmr_lib` or next to the 1H calculator first.
- [ ] Add **orchestration**: three converged `CNDO2SystemComplex` SCFs per axis ($-\varepsilon, 0, +\varepsilon$) with `set_magnetic_field(k, lambda)` + `fixed_point::solve_cndo`, then build $\partial P / \partial B_k$ from $P_\alpha + P_\beta$ (or spin-resolved if required).
- [ ] Implement **$\sigma^A_{p,iso}$** (paramagnetic isotropic shielding for nucleus $A$) via
  $$\sigma^A_{p,iso} = \frac{1}{3} \sum_{k \in \{x,y,z\}} Tr \left[ \frac{\partial P}{\partial B_k} H^{(1,1)}_{A,k} \right]$$
  using `compute_shielding_operator_matrix(k, A)` as $H^{(1,1)}_{A,k}$ (plus any global prefactor / ppm scaling from your source). *Pople’s closed form for the shielding tensor and its isotropic average is Eqs. (5.9)–(5.10); the underlying second-order energy identification uses Eq. (5.4) and the perturbation elements (5.5)–(5.8) substituted into Eq. (2.20).*
- [ ] Wire **`nmr_1H_calculator`**: replace placeholder δ table with $\sigma_{ref} - \sigma_{sample}$ once $\sigma(H)$ exists; align gauge with the perturbed SCF.

---

## Implemented: uniform field in the CNDO/2 Fock matrix (complex SCF)

The one-electron Hamiltonian in Pople’s setup is the minimal-coupling form **Eq. (2.1)** with vector potential **Eq. (2.2)** (uniform field plus nuclear dipole terms; for susceptibility he retains only the $\frac{1}{2} H \times r$ piece until Sec. V). MOs use gauge-invariant AOs **Eqs. (2.3)–(2.4)** and satisfy the LCAO secular problem **Eqs. (2.5)–(2.6)** with field-dependent matrix elements **Eqs. (2.7)–(2.8)**.

The complex electronic Hamiltonian/Fock build adds a **minimal-coupling-style** imaginary perturbation proportional to the **angular momentum** operator along the applied field direction $k \in \{x,y,z\}$ (same physical content as the first-order magnetic piece in **Eq. (2.17)**, where $(A - A_\mu) \cdot p$ is proportional to orbital angular momentum about nucleus $\mu$ for a uniform field):

$$
F = F^{(0)} + i \lambda L_k
$$

($\lambda$ small; same form for $\alpha$ and $\beta$ blocks in the current code.) SCF uses complex density matrices $P_\alpha, P_\beta$ and diagonalizes $F$ each iteration.

**Gauge origin** for building $L$ in AO integrals: **center of mass** of the molecule (implemented as `molecule_center_of_mass` and passed into the angular-momentum matrix builder used by the Fock template). *This addresses the “distant gauge origin” cancellation issue discussed around **Eq. (2.2)** and the London / GIAO construction **Eqs. (2.3)–(2.4)**; Pople’s screening theory (Sec. V) again uses the full **Eq. (2.2)** and local gauges **Eq. (5.2)**.*

---

## Implemented: AO angular momentum matrix $L_k$

For direction $k$, components $(L_k)_{\mu\nu}$ are built from Gaussian AO products using the textbook cross-product form in Cartesian factors (e.g. for $L_z$: terms like $(x-X)\partial_y - (y-Y)\partial_x$ relative to the chosen gauge origin $(X,Y,Z)$). Integrals use the existing contracted-Gaussian derivative and coordinate factors measured from that origin (`get_gaussian_derivative`, `multiply_by_coords`, etc.). *In Pople’s notation this is the operator tied to $(A - A_\mu) \cdot p$ in **Eq. (2.17)** (see text immediately preceding that equation).*

---

## Implemented: point-dipole shielding **operator** $H^{(1,1)}_{A,k}$ (Pople-style skeleton)

Section V develops NMR shifts using the **full** vector potential **Eq. (2.2)**; with the observing nucleus at the origin, the dipole field enters as in **Eq. (5.1)**. **Approximation F** replaces the local $(A - A_\nu)$ by a uniform field matching **Eq. (5.1)** at each atomic center **Eq. (5.2)**, with the induced “local field” components **Eq. (5.3)**. First-order perturbation matrix elements for AO pairs both on a neighbor atom $B \neq A$ carry dipole $|R_{AB}|^{-3}$-type weighting and angular-momentum structure as in **Eq. (5.7)** (second-order pieces **Eq. (5.8)**).

For target nucleus **$A$** (proton index) and field component **$k$**, the AO matrix is **block-sparse**: only **intra-center** blocks on neighbor atoms $B \neq A$, weighted by $|R_{AB}|^{-3}$:

$$
\left( H^{(1,1)}_{A,k} \right)_{\mu \in B, \nu \in B}
=
\frac{1}{|R_{AB}|^{3}}
\left( L_k^{(B)} \right)_{\mu \nu}
$$

where $L_k^{(B)}$ denotes the **full** $L_k$ matrix computed with **gauge origin at $R_B$** (second argument to `compute_angular_momentum_matrix`), then restricted to the AO span of atom $B$. The block for $B=A$ is omitted (on-site piece treated as zero in this approximation). Off-diagonal blocks between different atoms are zero.

*Code:* `CNDO2SystemComplex::compute_shielding_operator_matrix(k, target_proton_idx)`.

---

## Target: paramagnetic shielding (finite perturbation / Ramsey)

Pople identifies the second-order shielding energy with **Eq. (5.4)** and obtains the shielding tensor by comparing to the perturbation expansion; the **paramagnetic** (bond-order–type) contribution is traced through **Eq. (2.20)** with the magnetic matrix elements **Eqs. (5.5)–(5.8)** (see text following **Eq. (5.8)**). The design target below is a **finite-field / density-derivative** packaging of the same physics rather than Pople’s explicit sum-over-states bond-order form **Eqs. (5.9)–(5.10)**.

The **isotropic paramagnetic** term used as the design target:

$$
\sigma^A_{p,iso}
=
\frac{1}{3} \sum_{k \in \{x,y,z\}}
Tr \left(
\frac{\partial P}{\partial B_k}
H^{(1,1)}_{A,k}
\right)
$$

**Not yet implemented:** $\partial P / \partial B_k$ via **finite differences** of the **converged** SCF density (central or forward difference in the field parameter), then trace with $H^{(1,1)}_{A,k}$ above, overall prefactors / ppm conversion. *For the isotropic shielding constant in Pople’s fully reduced theory see **Eq. (5.10)** (full tensor **Eq. (5.9)**).*

---

## Related pieces (brief)

- **Real** `CNDO2System` SCF: still used for Mulliken-like populations (`get_electron_density`) and ¹³C shielding path; **complex** `CNDO2SystemComplex` carries the magnetic perturbation in Fock.
- **`nmr_1H_calculator`**: runs real + complex SCF and prints a **placeholder** δ table until $\sigma(H)$ is assembled from the pieces above.
