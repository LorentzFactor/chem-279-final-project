# HW 5 - Evaluating the Gradient of SCF Energy

#### Alex Chase 
achase95@berkeley.edu

## Overview

The purpose of this assignment was to apply the derivations for the analytical energy gradient from part 1 and actually calculate these gradients for several molecules. The original derivation can be found in `Part1Derivation.md` in this repository. The `Molecule` class was extended from previous assignments to include this functionality. Once the gradients were confirmed to be correct for the original test case, I implemented a steepest gradient descent optimizer to determine optimal geometric configurations of several molecules using the CNDO/2 with SCF method. The input files are located in the `optimize_input/` directory and the optimization can be ran using the executable generated from `opt_geometry.cpp`.

The geometries for the following molecules were optimized using this approach: CO, HF, H2O and NH3 (Table 1). There are several interesting results to note from this experiment. Firstly, the calculated bond lengths for CO and HF from this optimization are closer to the actual values compared to the Pople results. The Pople paper employed a coarser scan of bond lengths to find a minimum energy due to computational limitations of the time. In contrast, I was able to employ a more computationally expensive gradient descent approach which arrived at a more accurate solution for these diatomics. 

Secondly, the bond lengths and angles observed for H2O and NH3 illustrate the limitations of the CNDO/2 approach for some molecules. In the Pople paper, for these molecules the bond lengths were fixed and only the various bond angles were scanned. Once again this was due to computation restrictions of performing a gradient descent optimization. Therefore they arrive at a more accurate bond angle for NH3. In contrast we see that this codes optimization approach slightly over estimates bond lengths and in the case of NH3 underestimates the bond angle. 

Due to the approximations made by CNDO/2 to simplify the computation, implementing a gradient descent approach without fixed, empirical bond lengths allow for unrealistic geometries to be predicted for some molecules. In general though, this additional experiment shows that the CNDO/2 is relatively close to the actual geometries for these simple molecules and confirms the result that Pople observed when first publishing this approach.

|          | Actual           |                 | Pople CNDO/2     |                 | HW 5 CNDO/2      |                 |
|----------|------------------|-----------------|------------------|-----------------|------------------|-----------------|
| Molecule | Bond Angle (deg) | Bond Length (A) | Bond Angle (deg) | Bond Length (A) | Bond Angle (deg) | Bond Length (A) |
| CO       | -                | 1.128           | -                | 1.191           | -                | 1.155           |
| HF       | -                | 0.917           | -                | 1.000           | -                | 0.984           |
| H2O      | 104.5            | 0.958           | 107.1            | 0.960           | 104.0            | 1.014           |
| NH3      | 107.8            | 1.010           | 106.7            | 1.020           | 104.8            | 1.051           |

#### Table 1. Geometry Optimization Results