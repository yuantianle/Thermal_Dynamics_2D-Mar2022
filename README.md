# Thermal_Dynamics_2D-Mar2022

## Documentation

Please check: [2D Thermal Dynamics](https://yuantianle.github.io/1_Computer_Graphics/Science/Thermal/)

## Illustration for Five-point stencil method

|![chpt4_Five_stenclie_iter](https://github.com/yuantianle/Thermal_Dynamics_2D-Mar2022/assets/61530469/46427cb2-ffc1-4c20-908e-b5f9a97d21a5)|![chpt4_Five_stenclie_grad](https://github.com/yuantianle/Thermal_Dynamics_2D-Mar2022/assets/61530469/1996951d-bb0f-4d03-9ebe-9a2c25da8a93)|
|-|-|

## Iterating result

|Boundary Condition|Initial Temperature|Iteration till the stable|
|-|-|-|
|Dirichlet (top, right) Neumann (bottom, left)| High (top, right) Low (bottom, left) |![matrix](https://github.com/yuantianle/Thermal_Dynamics_2D-Mar2022/assets/61530469/4b681c22-e4c9-45bb-addc-3734cb80caeb)|
|Dirichlet (top) Neumann (bottom, left, right)| High (top) Low (bottom, left, right)|![matrix](https://github.com/yuantianle/Thermal_Dynamics_2D-Mar2022/assets/61530469/9d8dfd07-b233-4d59-9722-602d4b60caa9)|
|Dirichlet (top, bottom, left, right)| High (top, right) Low (bottom, left)|![matrix](https://github.com/yuantianle/Thermal_Dynamics_2D-Mar2022/assets/61530469/20012676-030b-47db-a319-ce53fc96b85b)|
|Dirichlet (top, bottom, left, right)| High (top) Low (bottom, left, right)|![matrix](https://github.com/yuantianle/Thermal_Dynamics_2D-Mar2022/assets/61530469/e34d50a1-40ae-41ba-b45e-b277f6aafd2f)|
