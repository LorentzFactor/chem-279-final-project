## Homework 5 - Part 1
#### Alex Chase

The purpose of the following derivation is to determine the analytical energy gradient for a CNDO/2 calculation. The negative of the energy gradient is the force acting upon different atoms in our molecule. In the next phase of this assignment we will be able to use this gradient to optimize our atomic coordinates to find the minimum energy state that optimizes the molecular geometry. In this derivation we will first determine the $x_{\mu\nu}$ component for the overlap term, followed by the $y_{AB}$ for the electron repulsion term, and lastly solve for the derivatives of contracted gaussians within $s$ and $p$ orbitals with respect to the nuclear center. We first examine the energy equation for CNDO/2.

$$E^{R_A}_{CNDO/2} = \sum_{\mu \neq \nu} x_{\mu \nu}s^{R_A}_{\mu \nu} + \sum_{A \neq B} y_{AB} \gamma^{R_A}_{AB} + V_{nuc}^{R_A}$$

$$E_{CNDO/2} = \frac{1}{2} \sum_{\mu \nu}p^{\alpha}_{\mu \nu}(h_{\mu \nu} + f^{\alpha}_{\mu \nu}) + \frac{1}{2} \sum_{\mu \nu}p^{\beta}_{\mu \nu}(h_{\mu \nu} + f^{\beta}_{\mu \nu}) + \sum_{A} \sum_{B < A} \frac{Z_A Z_B}{R_{AB}}$$

The off diagonal elements for fock and hamiltonian energies contain $s_{\mu \nu}$.

$$f^{\alpha}_{\mu \nu} = \frac{1}{2}(\beta_A + \beta_B)s_{\mu \nu} - p^{\alpha}_{\mu \nu}\gamma_{AB} $$

$$h_{\mu \nu} = \frac{1}{2}(\beta_A + \beta_B)s_{\mu \nu}$$

Substituting into the previous expression yields the following.

$$p^{\alpha}_{\mu \nu}(h_{\mu \nu} + f^{\alpha}_{\mu \nu}) = p^{\alpha}_{\mu \nu}\left[\frac{1}{2}(\beta_A + \beta_B)s_{\mu \nu} + \frac{1}{2}(\beta_A + \beta_B)s_{\mu \nu} - p^{\alpha}_{\mu \nu}\gamma_{AB}\right]$$

$$p^{\alpha}_{\mu \nu}[(\beta_A + \beta_B)s_{\mu \nu} - p^{\alpha}_{\mu \nu}\gamma_{AB}] $$

$$p^{\alpha}_{\mu \nu}(\beta_A + \beta_B)s_{\mu \nu} + (-(p^{\alpha}_{\mu \nu})^2)\gamma_{AB} $$

Evaluate $p^{\beta}_{\mu \nu}$ yields an equivalent expression.

$$p^{\beta}_{\mu \nu}(\beta_A + \beta_B)s_{\mu \nu} + (-(p^{\beta}_{\mu \nu})^2)\gamma_{AB} $$

The total density is the sum of the density on alpha and beta electrons.

$$p^{tot}_{\mu \nu} = p^{\alpha}_{\mu \nu} + p^{\beta}_{\mu \nu} $$

Inserting just the density terms containing $s_{\mu \nu}$ back into our $E_{CNDO/2}$ expression gives us the following.

$$E_{CNDO/2} = \frac{1}{2} \sum_{\mu \nu}p^{\alpha}_{\mu \nu}(h_{\mu \nu} + f^{\alpha}_{\mu \nu}) + \frac{1}{2} \sum_{\mu \nu}p^{\beta}_{\mu \nu}(h_{\mu \nu} + f^{\beta}_{\mu \nu})$$

$$\frac{1}{2} p^{\alpha}_{\mu \nu}(\beta_A + \beta_B)s_{\mu \nu} + \frac{1}{2} p^{\beta}_{\mu \nu}(\beta_A + \beta_B)s_{\mu \nu}$$

$$\frac{1}{2}(p^{\alpha}_{\mu \nu} + p^{\beta}_{\mu \nu})(\beta_A + \beta_B)s_{\mu \nu}$$

$$\frac{1}{2}p^{tot}_{\mu \nu}(\beta_A + \beta_B)s_{\mu \nu}$$

Looking back at our initial expression for $E^{R_A}_{CNDO/2}$ we can see that $x_{\mu \nu}$ equals the following. Additionally, because we are now summing over unique pairs as opposed to all off-diagonals, we multiply our previous expression by a factor of 2 when inserting into the following summation.

$$ x_{\mu \nu} =  p^{tot}_{\mu \nu}(\beta_A + \beta_B)$$  

$$ \sum_{\mu \neq \nu} x_{\mu \nu}s^{R_A}_{\mu \nu} =  \sum_{\mu \neq \nu} p^{tot}_{\mu \nu}(\beta_A + \beta_B) s^{R_A}_{\mu \nu}$$  

The diagonal elements of the fock and hamiltonian energies contain $\gamma_{AC}$ terms that we need to solve for $y_{AB}$ so we will evaluate those next.

$$f^{\alpha}_{\mu \mu} = -\frac{1}{2}(I_{\mu} + A_{\mu}) + \left[(p^{tot}_{AA} - Z_A) - \left(p^{\alpha}_{\mu \mu} - \frac{1}{2}\right)\right] + \sum_{C \neq A}(p^{tot}_{CC} - Z_C)\gamma_{AC} $$

$$h_{\mu \mu} =  -\frac{1}{2}(I_{\mu} + A_{\mu}) - \left(Z_A - \frac{1}{2}\right) - \sum_{C \neq A}Z_C\gamma_{AC} $$

Reducing this down to just focus on the $\gamma_{AC}$ terms.

$$f^{\alpha}_{\mu \mu} = \sum_{C \neq A}(p^{tot}_{CC} - Z_C)\gamma_{AC} $$

$$h_{\mu \mu} =  - \sum_{C \neq A}Z_C\gamma_{AC} $$

$$p^{\alpha}_{\mu\mu}(h_{\mu\mu} + f^{\alpha}_{\mu\mu}) = p^{\alpha}_{\mu\mu}\left[ - \sum_{C \neq A}Z_C\gamma_{AC} +  \sum_{C \neq A}(p^{tot}_{CC} - Z_C)\gamma_{AC}\right] $$

$$p^{\alpha}_{\mu\mu}(h_{\mu\mu} + f^{\alpha}_{\mu\mu}) = p^{\alpha}_{\mu\mu}(p^{tot}_{CC} - 2Z_C)\gamma_{AC}  $$

$$p^{\beta}_{\mu\mu}(h_{\mu\mu} + f^{\beta}_{\mu\mu}) = p^{\beta}_{\mu\mu}(p^{tot}_{CC} - 2Z_C)\gamma_{AC}  $$

$$p^{tot}_{\mu\mu} = p^{\alpha}_{\mu\mu} + p^{\beta}_{\mu\mu} $$

$$(p^{\alpha}_{\mu\mu} + p^{\beta}_{\mu\mu})( p^{tot}_{CC} - 2Z_C)\gamma_{AC} $$ 

$$p^{tot}_{\mu\mu}( p^{tot}_{CC} - 2Z_C)\gamma_{AC} $$ 

Our sum for all $\mu$ on a particular Atom only depends on $p^{tot}_{\mu\mu}$. Therefore our first term for $E_{CNDO/2}$ looks like the following.

$$\frac{1}{2}\left(\sum_{\mu\in A} p^{tot}_{\mu\mu}\right)(p^{tot}_{CC} - 2Z_C)\gamma_{AC}  $$

This summation is equal to the total density for atom A. We are comparing it against its neighbor B, which replaces C in the following expression.

$$\frac{1}{2}p^{tot}_{AA}(p^{tot}_{BB} - 2Z_B)\gamma_{AB}  $$

Similarly when our current atom is B we get the following.

$$\frac{1}{2}p^{tot}_{BB}(p^{tot}_{AA} - 2Z_A)\gamma_{BA}  $$

$$ \gamma_{AB} = \gamma_{BA} $$

Adding the expressions for atoms A and B gives us the $y_{AB}$ contribution from the diagonal elements.

$$\frac{1}{2}p^{tot}_{AA}(p^{tot}_{BB} - 2Z_B)\gamma_{AB} + \frac{1}{2}p^{tot}_{BB}(p^{tot}_{AA} - 2Z_A)\gamma_{AB} $$

$$\left(\frac{1}{2}p^{tot}_{AA}p^{tot}_{BB} - p^{tot}_{AA}Z_B\right)\gamma_{AB} + \left(\frac{1}{2}p^{tot}_{BB}p^{tot}_{AA} - p^{tot}_{BB}Z_A\right)\gamma_{AB}$$

$$(p^{tot}_{AA}p^{tot}_{BB} - Z_Bp^{tot}_{AA} - Z_Ap^{tot}_{BB})\gamma_{AB} $$

Finally adding in our terms that we previously derived from the off diagonal elements we get the combined expression.

$$(p^{tot}_{AA}p^{tot}_{BB} - Z_Bp^{tot}_{AA} - Z_Ap^{tot}_{BB} - \sum_{\mu\in A} \sum_{\nu\in B}((p^{\alpha}_{\mu\nu})^2 + (p^{\beta}_{\mu\nu})^2))\gamma_{AB} $$

$$(p^{tot}_{AA}p^{tot}_{BB} - Z_Bp^{tot}_{AA} - Z_Ap^{tot}_{BB} - \sum_{\mu\in A} \sum_{\nu\in B}(p^{\alpha}_{\mu\nu}p^{\alpha}_{\nu\mu} + p^{\beta}_{\mu\nu}p^{\beta}_{\nu\mu}))\gamma_{AB} $$

$$y_{AB} = p^{tot}_{AA}p^{tot}_{BB} - Z_Bp^{tot}_{AA} - Z_Ap^{tot}_{BB} - \sum_{\mu\in A} \sum_{\nu\in B}(p^{\alpha}_{\mu\nu}p^{\alpha}_{\nu\mu} + p^{\beta}_{\mu\nu}p^{\beta}_{\nu\mu}) $$

Now that we have our $x_{\mu\nu}$ and $y_{AB}$ terms we can find the derivative of contracted gaussian $s$ orbital with respect to its nuclear center. The derivative of the overlap with respect to a particular direction $X$ can be written as the following.

$$\frac{\partial S^{kl}}{\partial X_{a}} = -l_kS_{1D}(X_A, X_B, l_k - 1, l_l, \alpha_k, \alpha_l) + 2\alpha_k S_{1D}(X_A, X_B, l_k + 1,l_l, \alpha_k, \alpha_l) $$

For an $s$ orbital the angular momentum is 0 in all directions. We can write the contracted gaussian overlap on atom A as the following with $g_k(r)$ as our contracted gaussian.

$$\mu(r) = \sum^{3}_{k=1}d_{k\mu}N_{k\mu}g_k(r) $$

The derivative with respect to the X-dimension is then the following.

$$\frac{\partial \mu_s}{\partial X_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(-(0)S_{1D}(X_A, X_B, (0) - 1, l_l,\alpha_k, \alpha_l) + 2\alpha_k S_{1D}(X_A, X_B, (0) + 1,l_l, \alpha_k, \alpha_l)) $$

$$\frac{\partial \mu_s}{\partial X_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(2\alpha_k S_{1D}(X_A, X_B, 1, l_l, \alpha_k, \alpha_l)) $$

The derivatives with respect to the $y$ and $z$ dimensions are the same for the $s$ orbital. Now we can do the same thing for our $p$ orbitals. Using $p_x$ as an example we can find the partial derivatives with respect to the $x, y, z$ dimensions.

$$ p_x(l_x = 1, l_y = 0, l_z = 0) $$

With respect to X dimension.

$$\frac{\partial \mu_{p_x}}{\partial X_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(-(1)S_{1D}(X_A, X_B, (1) - 1,l_l, \alpha_k, \alpha_l) + 2\alpha_k S_{1D}(X_A, X_B, (1) + 1,l_l, \alpha_k, \alpha_l)) $$

$$\frac{\partial \mu_{p_x}}{\partial X_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(-S_{1D}(X_A, X_B, 0, l_l, \alpha_k, \alpha_l) + 2\alpha_k S_{1D}(X_A, X_B, 2,l_l, \alpha_k, \alpha_l)) $$

With respect to the Y dimension.

$$\frac{\partial \mu_{p_x}}{\partial Y_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(-(0)S_{1D}(Y_A, Y_B, (0) - 1, l_l, \alpha_k, \alpha_l) + 2\alpha_k S_{1D}(Y_A, Y_B, (0) + 1,l_l, \alpha_k, \alpha_l)) $$

$$\frac{\partial \mu_{p_x}}{\partial Y_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(2\alpha_k S_{1D}(Y_A, Y_B, 1, l_l, \alpha_k, \alpha_l)) $$

And lastly the same as Y with respect to the Z dimension.

$$\frac{\partial \mu_{p_x}}{\partial Z_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(2\alpha_k S_{1D}(Z_A, Z_B, 1, l_l, \alpha_k, \alpha_l)) $$

The same logic then applies for the $p_y$ and $p_z$ orbitals.

### Results Summary

$$ x_{\mu \nu} =  p^{tot}_{\mu \nu}(\beta_A + \beta_B)$$  

$$y_{AB} = p^{tot}_{AA}p^{tot}_{BB} - Z_Bp^{tot}_{AA} - Z_Ap^{tot}_{BB} - \sum_{\mu\in A} \sum_{\nu\in B}(p^{\alpha}_{\mu\nu}p^{\alpha}_{\nu\mu} + p^{\beta}_{\mu\nu}p^{\beta}_{\nu\mu}) $$

$$\frac{\partial \mu_s}{\partial X_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(2\alpha_k S_{1D}(X_A, X_B, 1, l_l, \alpha_k, \alpha_l)) $$

$$\frac{\partial \mu_{p_x}}{\partial X_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(-S_{1D}(X_A, X_B, 0, l_l, \alpha_k, \alpha_l) + 2\alpha_k S_{1D}(X_A, X_B, 2, l_l, \alpha_k, \alpha_l)) $$

$$\frac{\partial \mu_{p_x}}{\partial Y_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(2\alpha_k S_{1D}(Y_A, Y_B, 1, l_l, \alpha_k, \alpha_l)) $$

$$\frac{\partial \mu_{p_x}}{\partial Z_A} = \sum^3_{k=1}d_{k\mu}N_{k\mu}(2\alpha_k S_{1D}(Z_A, Z_B, 1, l_l, \alpha_k, \alpha_l)) $$