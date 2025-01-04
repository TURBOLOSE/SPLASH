# SPLASH: SPreading LAyer Simulation for Hydrodynamics

Article preprint avaliable at https://arxiv.org/abs/2412.00867

Credit to https://github.com/pmp-library/pmp-library and Daniel Sieger (https://danielsieger.com/blog/2021/03/27/generating-spheres.html) for mesh related headers.
Credit to https://github.com/nlohmann/json for json processing.

This repo contains work in progress version of a MUSCL scheme for gas dynamics on an unstructured spherical mesh. 
It is designed to solve the problem of a spreading layer of a neturon star but it can be modified to solve other problems.
The article describing all the mehods used is also still in progress.

## Getting started

Attention: This project is dependent on the library `Eigen` that you can get from [official page](https://eigen.tuxfamily.org/index.php?title=Main_Page).
You do not need to install library `pmp`, all the headers necessary are included in this repo.

Include any Riemann solver header from src/Riemann_solvers, for example **"src/Riemann_solvers/HLLCplus.hpp"**.
Then you need to select a spherical mesh. You have 4 main meshes, `uv_sphere(nx,ny)`,`quad_sphere(n)`,`icosphere(n)` and `icosphere_hex(n)`.
Here `nx,ny` -- numbers of cols and rows of a UV mesh, `n`--number of recursive subdivisions of a mesh based on a cube, icosahedron or dodecahedron.
Be carefull as the number `n` does not represent the amount of faces you can expect. For example `icosphere_hex(5)` consists of ~10,000 faces.

![meshes](https://github.com/TURBOLOSE/MUSCL-scheme-on-spherical-mesh-WIP/assets/129312616/90ebcd68-58cc-4d32-9901-dc25b40b90a6)

Then you can create an instance of a solver using ` MUSCL_HLLCplus solver(mesh, U_in, dim, gam, omega_ns);`, 
where `mesh` -- you selected mesh, `U_in` -- vector of input values (you can generate them using a python scipt in the repo), `dim` -- number of dimensions of your system of equations (usually 5 or 4),
`gam` -- 2 dimensional adiabatic gamma, `omega_ns` -- angular velocity of a surface underneath (set to 0 for no rotation).

To compile the code run the makefile in the directory. In the future code will be adapted to different compilers with use of CMake or Premake.

### Methods of the MUSCL_(*Riemann solver*) class. 
Use `do_step(dt)` to integrate a single step in time with a fixed step size. Be aware that inside algorithm may reduce step size in order to preserve stability.

* `time()` returns `double` current time of an integrator.
* `get_stop_check()` returns `bool` that turns `True` if the time stepping needs to be stopped, usually due to the loss of stability or an appearence of `NaN`.
* `write_face_centers()` creates a `face_centers.dat` file in the results folder that contains the cartesian coordinates of all face centers of a mesh. 
 These are the coodrinates on a polyhedron, not on a sphere, their radius is less than 1.
 * `write_vertices()` creates a `vertices.dat` file in the results folder that contains the cartesian coordinates of all vertices of a mesh.
 These points are locate on a sphere and their radius is equal to 1.
 * `write_faces()` reates a `faces.dat` file in the results folder that contains the indexes of a vertices that make a face.
 Different faces are located in a different lines in a file. For example, a line 1 3 7 means that the face is a triangle made of vertices with numbers 1,3 and 7 (numeration starts with 0).
 * `write_t_rho()` adds a new line in a file `rho.dat` located in the folder results. Each line starts with the current time of integration `t` and continues with all density values in the center of each face.
 You should use this in a loop over time.
 * `write_t_p()` same, but writes pressure into `p.dat`.
 * `write_t_curl()` same, but writes curl into `curl.dat`.
 * `write_t_omega_z()` same, but writes z component of an angular velocity of each face into `omega.dat`.

 You can find a working loop inside of main.cpp.

 ### Input
 The main input of a programm consists of 2 parts: the initial data stored in `input/input.dat` and the integration parameters stored in `input/parameters.json`. The initial data file can be created using functions present in the `make_input.py` file. Any of them will generate input.dat file for the mesh that is stored in the `results` directory.
 
 If no files are present there or if you want to change the mesh you need to run a C++ code up to the point when it says "processing mesh done". It means that files containing mesh data (`results/face_centers.dat`, `results/faces.dat`, `results/vertices.dat`) were generated and can be used by a python code to generate a file with the initial state or create plots after the integration.

 `parameters.json` contains several integration parameters:
 * `dim` — the number of dimensions that should be either 4 or 5 standing for isothermal and adiabatic physics respectively.
 * `dt` — the upper time step limit. It may become lower during the integration to preserve stability.
 * `t_max` — full simulation time (in code units)
 * `maxstep` — the number of steps to compute. The simulation will stop either at `t_max` or when the number of steps reaches `maxstep` (whichever happens first).
 * `skipstep` — the number of steps to skip while writing data into files.
 * `gam3d` — the adiabatic index (corresponds to `\gamma` in the article). `gam2d=2-1/gam3d`.
 * `omega_ns` — the angular velocity of initial rotation of a sphere. (usually should be set to 0, as the source terms are somewhat unstable).
 * `threads` — the number of openmp threads to use.
 * `accretion_on` — boolean, enables or disables accretion.
 * `accretion_rate` — full accretion rate in code units.
 * `e_acc` — $\Pi_{acc} / \Sigma_{acc}$ in code units.
 * `omega_acc_abs` — rotational frequency of the accreted matter in $z_{acc}$ axis.
 * `tilt_angle` — defines tilt of $z_{acc}$ axis in relation to $z$ axis.
 * `acc_width` — width of an accretion zone in $R_{ns}$. Accretion zone is distributed normally by latitude.

 
 
### Visualisation of results.
You can use python script `plots.py` to create plots of the values recorded. Example of a plot:

![fig0011](https://github.com/TURBOLOSE/MUSCL-scheme-on-spherical-mesh-WIP/assets/129312616/e986f42c-cb2e-4af0-819b-3a204be2fb5e)


These plots can be turned into a video. Example of a long simulation (click on gif to go to YouTube):



[![IMAGE ALT TEXT HERE](https://i.ytimg.com/an_webp/9_TPXbtqDNY/mqdefault_6s.webp?du=3000&sqp=CPTi5boG&rs=AOn4CLC5WTxrNHvtzkzG0bWaYl96htBVGA)](https://www.youtube.com/watch?v=9_TPXbtqDNY)

## C++ class structure

The code is divided into multiple classes:
* `MUSCL_geometry` processes mesh and contains all coefficients needed for geometric part of integration.
* `MUSCL_base` is a virtual class that is responsible for timestepping. It has to be used to create a derived class with functions `flux_star` — Riemann solver flux, `limiter` — MUSCL method flux limiter, `source` — source term of the system. (As of right now, `MUSCL_base` also converts energy into pressure in adiabatic case, so if you want to change the way pressure or energy are calculated, you should also imlpement changes inside this class).
* `adiabatic` and `isothermal` are derived from  `MUSCL_base`. They are also virtual but they only need `flux_star`. These classes define the physics, structure of fluxes and source terms, as well as flux limiters.
* `HLLE`, `HLLE_p`, `HLLC`, `HLLC+` are the classes that define `flux_star` function. 


![classes](https://github.com/TURBOLOSE/MUSCL-scheme-on-spherical-mesh-WIP/assets/129312616/41918eb6-6dee-4481-8654-af82b6d903ed)

## Units used in code


* $\gamma = 4/3$
* $\Gamma = 2-1/\gamma=5/4$
* $R_{unit}=10 km $
* $V_{unit} = 1 \rm c$
* $\Sigma_{unit} = 10^7 g/cm^2$
* $t_{unit} = R_{unit}/V_{unit} = 3.33 \cdot 10^{-5} s$
* $\Pi_{unit} = 9 \cdot 10^{27} g \ s^{-2}$


