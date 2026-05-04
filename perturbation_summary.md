# Proton NMR perturbation — theory, implementation, calibration, and how to run

**Primary reference (equation numbers below):** J. A. Pople, “Molecular-Orbital Theory of Diamagnetism. I. An Approximate LCAO Scheme,” *J. Chem. Phys.* **37**, 53 (1962) (DOI [10.1063/1.1732974](https://doi.org/10.1063/1.1732974)); local copy `pople1962.pdf`.

This note is for the **current** CNDO/2 + complex-SCF **¹H shielding** path: what is implemented in code, how **empirical δ** is obtained via **linear regression**, and how to **run** the C++ binaries and Python helpers.

---

## Implemented at a glance

- [x] **Uniform field in the complex Fock matrix** — minimal-coupling-style term $F = F^{(0)} + i\lambda L_k$ with $k$ along x, y, or z (`set_magnetic_field`, `fock.tpp`).
- [x] **AO angular momentum** $L_k$ with **gauge origin at the molecular center of mass** for the Fock build.
- [x] **Shielding operator** $H^{(1,1)}_{A,k}$ for target proton $A$: neighbor-only intra-atomic blocks, $|R_{AB}|^{-3}$-weighted $L_k$ with **gauge origin at neighbor** $B$ (`compute_shielding_operator_matrix`).
- [x] **Paramagnetic shielding tensor** via **finite differences** of the **converged complex SCF density** in $\lambda$, contracted with $H^{(1,1)}$ (`compute_proton_shielding_tensor` + `nmr_lib::central_difference_density_derivative_wrt_B`).
- [x] **Isotropic trace**, **ppm scaling**, optional **local diamagnetic** term from real-SCF Mulliken-like $\rho_H$ on the 1s (`nmr_1H_calculator`, `nmr_1h_training_export`).
- [x] **Chemical shift**: either **uncalibrated** $\delta \approx \bar{\sigma}_{\mathrm{ref}} - \sigma_{\mathrm{total,main}}$ or **empirical linear map** from $\sigma_{\mathrm{para}}$ (JSON calibration).
- [x] **Training export + OLS** to fit calibration coefficients from a multi-molecule CSV (`nmr_1h_training_export`, `fit_nmr_shift_calibration.py`).

Remaining refinements (not blocking the pipeline): tightening the **$\lambda$ vs. $B$** bookkeeping if you need strict SI field amplitudes; optional **symmetrization** or alternate isotropic formulas for the numerical $\sigma_{pq}$ grid; extending beyond the current **Pople-style skeleton** for $H^{(1,1)}$.

---

## Complex CNDO/2 Fock and gauge (unchanged idea)

The one-electron Hamiltonian in Pople’s setup is the minimal-coupling form **Eq. (2.1)** with vector potential **Eq. (2.2)**. MOs use gauge-invariant AOs **Eqs. (2.3)–(2.4)** and the LCAO secular problem **Eqs. (2.5)–(2.6)** with field-dependent matrix elements **Eqs. (2.7)–(2.8)**.

The **complex** electronic Fock build adds an imaginary perturbation proportional to **angular momentum** along the chosen axis $k$:

$$
F = F^{(0)} + i\lambda L_k
$$

($\lambda$ is a small **dimensionless probe** in code; same form for $\alpha$ and $\beta$ blocks.) SCF uses complex density matrices $P_\alpha$, $P_\beta$ and **DIIS** (`diis::solve_cndo`).

**Gauge origin** for building $L$ in AO integrals used inside the Fock: **center of mass** of the molecule (`molecule_center_of_mass`).

---

## Shielding operator $H^{(1,1)}_{A,k}$ (Pople-style skeleton)

Section V of Pople (1962) motivates NMR shifts from the full vector potential; **Approximation F** uses a uniform induced field at each atomic center **Eq. (5.2)** with local-field components **Eq. (5.3)**. For target nucleus **$A$** (proton index) and field component **$k$**, the AO matrix is **block-sparse**: only **intra-center** blocks on neighbor atoms $B \neq A$, weighted by $|R_{AB}|^{-3}$:

$$
\left( H^{(1,1)}_{A,k} \right)_{\mu \in B,\,\nu \in B} = \frac{1}{|R_{AB}|^{3}} \left( L_k^{(B)} \right)_{\mu \nu}
$$

where $L_k^{(B)}$ is the full $L_k$ matrix with **gauge origin at $R_B$**, restricted to the AO span of atom $B$. The block for $B=A$ is omitted.

*Code:* `CNDO2SystemComplex::compute_shielding_operator_matrix(k, target_proton_idx)`.

---

## New: finite-difference density derivative and $\sigma$ tensor

Pople’s paramagnetic piece can be written in a **Ramsey / response** form using the field derivative of the density. The **design implemented in code** is:

1. For each Cartesian field axis **dir** in `{0,1,2}` (`compute_proton_shielding_tensor`):
   - Converge complex SCF with $F = F^{(0)} + i\lambda L_{\mathrm{dir}}$ at $\lambda = +\varepsilon$.
   - Reset densities, converge again at $\lambda = -\varepsilon$.
   - Form a **real** matrix approximating the first-order response of $P$ using the **imaginary** parts of the total density $P = P_\alpha + P_\beta$:

$$
\frac{\partial P}{\partial \lambda} \approx \frac{\mathrm{Im}\,P(\lambda=+\varepsilon) - \mathrm{Im}\,P(\lambda=-\varepsilon)}{2\varepsilon}
$$

   *Implementation:* `nmr_lib::central_difference_density_derivative_wrt_B` (name reflects the physical goal $\partial P / \partial B$; the finite difference is taken in the **code parameter** $\lambda$).

2. For each **response / operator direction** $j \in \{0,1,2\}$, build $H^{(1,1)}_{A,j}$ and accumulate (field axis from step 1 is $k$):

$$
\sigma_{jk} = \mathrm{Tr}\left( \frac{\partial P}{\partial \lambda} \, H^{(1,1)}_{A,j} \right)
$$

yielding a **3×3** numerical tensor (`sigma_tensor` in code; entries $\sigma_{jk}$).

3. **Isotropic paramagnetic contribution (atomic units)** uses the **matrix trace** of that 3×3 tensor, divided by 3 (same convention as `nmr_1H_calculator` / training export):

$$
\sigma_{p,\mathrm{iso}}^{(\mathrm{raw})} = \frac{1}{3}\,\mathrm{Tr}(\mathbf{\sigma})
$$

4. **ppm scaling (default)** matches the codebase convention:

$$
\sigma_{\mathrm{para,ppm}} = \sigma_{p,\mathrm{iso}}^{(\mathrm{raw})} \cdot \alpha^2 \cdot 10^{6}, \quad \alpha^{-1} = 137.035999084
$$

5. **Optional diamagnetic proton term** (local, population-based):

$$
\sigma_{\mathrm{dia,ppm}} = \rho_H \cdot \frac{\alpha^2 \cdot 10^{6}}{3}, \quad \sigma_{\mathrm{total}} = \sigma_{\mathrm{para,ppm}} + \sigma_{\mathrm{dia,ppm}}
$$

where $\rho_H$ is the **real** CNDO/2 Mulliken-like electron density on the hydrogen’s basis (`get_electron_density`). You can disable the dia term in the training JSON with `"include_dia": false`.

*Code:* `CNDO2SystemComplex::compute_proton_shielding_tensor`, `nmr_1h_training_export.cpp`, `nmr_1H_calculator.cpp`.

**Orchestration note:** `nmr_1H_calculator` first builds **field-free** and **small fixed-$\lambda_{\mathrm{probe}}$** complex systems along a configurable **`field_dir`** (default $z$) so the unperturbed MO landscape exists for the shielding routine; **`compute_proton_shielding_tensor`** then performs its **own** axis sweeps at $\pm\varepsilon$ per proton. Inner SCF loops use DIIS with zeroed initial densities for each finite-difference solve.

---

## Chemical shift in `nmr_1h_calc`

**Uncalibrated (default)** — reference TMS (or other molecule) averaged shielding vs main:

$$
\delta_{\mathrm{ppm}} \approx \bar{\sigma}_{\mathrm{ref,total}} - \sigma_{\mathrm{main,total}}
$$

**Calibrated** — if both `shift_calibration_beta0` and `shift_calibration_beta1` appear in the **main** or **reference** JSON:

$$
\delta_{\mathrm{ppm}} = \beta_0 + \beta_1 \, \sigma_{\mathrm{para,main,ppm}} \quad \text{(absolute mode)}
$$

If `"shift_calibration_relative": true`:

$$
\delta_{\mathrm{ppm}} = \beta_1 \left( \sigma_{\mathrm{para,main}} - \langle \sigma_{\mathrm{para,ref}} \rangle \right)
$$

($\langle\cdot\rangle$ is the mean over reference hydrogens.)

---

## Training set, CSV export, and linear regression (OLS)

**Goal:** map computed **paramagnetic ppm feature** $\sigma_{\mathrm{para,ppm}}$ to experimental **¹H shifts** $\delta_{\mathrm{exp}}$ (literature / your spectra), because the raw CNDO/2 shielding scale does not match experiment without a **linear empirical correction**.

**Steps**

1. **Prepare a training list JSON** (see `sample_input/nmr_1h_training_export.json`):
   - `molecules[]`: each entry has `atoms_file_path`, electron counts, optional `distance_unit`, optional `label`, and **`experimental_shift_ppm`** — an array with **one value per hydrogen**, in the same order as hydrogens are discovered (**increasing global atom index**), matching `nmr_1h_training_export`.
   - Optional keys: `basis_dir`, `field_dir`, `epsilon`, `para_ppm_factor`, `dia_ppm_factor`, `include_dia`.
   - `output_csv`: where to write the feature table.

2. **Run `nmr_1h_training_export`** — for each molecule: real SCF (DIIS), complex SCF at $\lambda=0$, then for each H row **one** call to `compute_proton_shielding_tensor` (expensive). CSV columns include `sigma_para_ppm`, `sigma_dia_ppm`, `sigma_total_ppm`, `rho_H`, `delta_exp_ppm`.

3. **Fit ordinary least squares (OLS)** in Python on all rows with non-empty `delta_exp_ppm`:

$$
\delta_{\mathrm{exp}} \approx \beta_0 + \beta_1 \, \sigma_{\mathrm{para,ppm}}
$$

Closed form (same as `scripts/fit_nmr_shift_calibration.py`). Let $S_x = \sum_i x_i$, $S_y = \sum_i y_i$, etc., with $x_i = \sigma_{\mathrm{para,ppm}}$ and $y_i = \delta_{\mathrm{exp}}$:

$$
\beta_1 = \frac{n\,S_{xy} - S_x S_y}{n\,S_{xx} - S_x^2}
$$

$$
\beta_0 = \frac{1}{n}\left(S_y - \beta_1 S_x\right)
$$

4. **Copy** `shift_calibration_beta0` / `shift_calibration_beta1` into your per-molecule JSON files (see `sample_input/ethane.json` etc.) for **`nmr_1h_calc`**.

5. **Optional QA:** overlay predicted vs experimental sticks with `scripts/plot_nmr_training_spectrum.py` (uses the same linear map for predicted δ).

---

## How to run — build

From the repository root (adjust `build` if you use another directory):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Main executables: `build/nmr_1h_calc`, `build/nmr_1h_training_export`.

If you use the provided helper:

```bash
./build.sh
```

---

## How to run — `nmr_1h_calc` (shifts vs reference)

Two JSON configs: **main** molecule, then **reference** (commonly TMS).

```bash
./build/nmr_1h_calc sample_input/ethane.json sample_input/TMS.json
```

With custom build directory:

```bash
cmake --build /path/to/build -j4
/path/to/build/nmr_1h_calc sample_input/n_butane.json sample_input/TMS.json
```

Output: brief **progress lines** while shielding tensors are computed, then reference mean shielding, **per-proton** $\sigma$ and $\delta$, and an optional **grouped** summary when `shift_grouping_tol_ppm` on the main JSON is positive (set to 0 or below to disable).

---

## How to run — training export + fit + optional plot

**Export features CSV** from a training list:

```bash
./build/nmr_1h_training_export sample_input/nmr_1h_training_export.json
```

**Fit calibration** (reads `delta_exp_ppm` and `sigma_para_ppm` from the CSV):

```bash
python3 scripts/fit_nmr_shift_calibration.py student_output/nmr_1h_training_features.csv
```

**End-to-end shell pipeline** (export then fit; pass optional custom training JSON as first argument):

```bash
./scripts/run_nmr_calibration_pipeline.sh
./scripts/run_nmr_calibration_pipeline.sh path/to/my_training_list.json
```

**Plot** predicted vs training δ (requires matplotlib; see `python/requirements-viz.txt` if you use a venv):

```bash
pip install -r python/requirements-viz.txt
python3 scripts/plot_nmr_training_spectrum.py \
  sample_input/nmr_1h_training_export.json \
  student_output/nmr_1h_training_features.csv \
  --calibration-json sample_input/methane.json \
  --molecule n-butane \
  --out student_output/nmr_n_butane.png
```

Override coefficients without a JSON:

```bash
python3 scripts/plot_nmr_training_spectrum.py \
  sample_input/nmr_1h_training_export.json \
  student_output/nmr_1h_training_features.csv \
  --beta0 -2.2607 --beta1 10.1102 \
  --molecule ethane \
  --out student_output/nmr_ethane.png
```

---

## Related pieces (13C and legacy σ\_p)

- **Real** `CNDO2System` SCF: Mulliken-like populations for $\rho_H$ and the existing **¹³C** σ\_d / σ\_p (`nmr_lib`) path.
- **Complex** `CNDO2SystemComplex`: magnetic perturbation in the Fock matrix and **¹H** shielding tensor assembly described above.
