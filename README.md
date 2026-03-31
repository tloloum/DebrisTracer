# DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging

This repository contains the code and instructions for downloading the data used for the paper *DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging*

These instructions have been tested with a fresh install of Ubuntu 24.04 LTS. Adjustments may be required for other operating systems.


## Summary

- [Installation](#installation)

- [Example](#running-an-example)

- [Database](#downloading-the-database)

- [Reference](#reference)

## Installation

1. Download this repository  or run
   
```
 git clone https://github.com/tloloum/DebrisTracer.git
```

2. At the root of the repository, run (it can take a while)

```
./install.sh
```
3. To setup the the paraview and ttk environment variables in the current shell, run
```
export PATH=$PATH:$(pwd)/paraview/install/bin
export LD_LIBRARY_PATH=$(pwd)/paraview/install/lib:$(pwd)/ttk/install/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/paraview/install/lib/python3.12/site-packages:$(pwd)/ttk/install/lib/python3.12/site-packages
export PV_PLUGIN_PATH=$(pwd)/ttk/install/bin/plugins/TopologyToolKit

```
4. To permanently add the paraview and ttk environment variables to your .bashrc, run
```
echo "export PATH=\$PATH:$(pwd)/paraview/install/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$(pwd)/paraview/install/lib:$(pwd)/ttk/install/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)/paraview/install/lib/python3.12/site-packages:$(pwd)/ttk/install/lib/python3.12/site-packages" >> ~/.bashrc
echo "export PV_PLUGIN_PATH=$(pwd)/ttk/install/bin/plugins/TopologyToolKit" >> ~/.bashrc

```
and then  
```
source ~/.bashrc
```
to apply the changes. These commands should be called at the root of the repository.


5. At this point, you can launch paraview from the command line with 

```
paraview
```
or use the following scripts to automatically run the examples.

## Running an example
To run the provided example :

1. Download the reference dataset : [90° impact (1) (344 MB)](https://zenodo.org/records/19334114/files/HE0903.vti) and move it to the 'example' directory.

2. Go to the `example` directory
 
3. Next, from there, enter the following command:
```
paraview example.pvsm
```

This reproduces the images from the Figure 1 of the manuscript *DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging*. Note that Figure 1 is spread across 3 layouts in the example.

## Downloading the database
Database of hypervelocity impact acquisitions, capturing different dynamic fragmentation modalities (projectile-based and laser-based). These datasets are intended to be used with the accompanying processing software, available at [https://github.com/tloloum/DebrisTracer](https://github.com/tloloum/DebrisTracer).

The experimental tests were conducted on several facilities:
- A two-stage light-gas gun (2SLGG) named MICA [[1]](#ref1), located at ENSTA, Brest, France, and a 2SLGG HERMES at Thiot Ingenierie [[3]](#ref3). Both of these facilities are capable of accelerating a millimeter size projectile to hypersonic speeds between a few hundreds m/s and 6700 m/s, enabling studies of hypervelocity impacts.
- A high power laser facility named BELENOS [[2]](#ref2) at ENSTA, Brest, France. BELENOS laser pulse is capable of delivering a maximum energy of 3 J at 1064 nm with a Gaussian temporal pulse and a full width at half maximum of 7.5 ns.

In all experiments, transverse shadowgraphy with a white light source and ultra-high speed camera (Shimadzu HPVX2) up to 2 million frames per second was performed to visualize the ejected debris on the target rear face (opposite to the projectile impact / laser beam). The debris can be collected using a paraffin gel collector with a very high collection rate. Post-mortem gel samples were then analyzed by means of X-ray micro-tomography at CRT Morlaix (France). These analyses enabled the representative size and mass distribution of the ejecta to be determined and compared with values extracted from our Topological Data Analysis method (DebrisTracer).

- [90° impact (2) (463 MB)](https://zenodo.org/records/19334114/files/90_HE0905.vti)
- [90° impact (3) (546 MB)](https://zenodo.org/records/19334114/files/90_HE09013.vti)
- [90° impact (4) (558 MB)](https://zenodo.org/records/19334114/files/90_HE0904.vti)
- [90° impact (5) (67 MB)](https://zenodo.org/records/19334114/files/90_impact_M.vti)
- [45° impact (1) (216 MB)](https://zenodo.org/records/19334114/files/45_HE0911.vti)
- [45° impact (2) (224 MB)](https://zenodo.org/records/19334114/files/45_HE908.vti)
- [Laser impact (204 MB)](https://zenodo.org/records/19334114/files/laser_impact.vti)

## Reference

DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging

<a id="ref1"></a>[1] Seisson, G., Hébert, D., Hallo, L., Chevalier, J.-M., Guillet, F., 
Berthe, L., & Boustie, M. (2014). Penetration and cratering experiments of graphite by 
0.5-mm diameter steel spheres at various impact velocities. *International Journal of 
Impact Engineering*, 70, 14–20. [https://doi.org/10.1016/j.ijimpeng.2014.03.004](https://doi.org/10.1016/j.ijimpeng.2014.03.004)

<a id="ref2"></a>[2] Reynier, B., Mircioaga, R. M., Le Clanche, J., Taddei, L., Chevalier, 
J. M., Hébert, D., & Arrigoni, M. (2025). High-velocity laser-driven flyer impact on 
paraffin gel. *International Journal of Impact Engineering*. 
[https://doi.org/10.1016/j.ijimpeng.2025.105311](https://doi.org/10.1016/j.ijimpeng.2025.105311)

<a id="ref3"></a>[3] Thiot Ingénierie (2026). 
[https://www.thiot-ingenierie.com/fr/lanceurs-a-gaz-et-autres-equipements/lanceurs-a-gaz-double-etage/](https://www.thiot-ingenierie.com/fr/lanceurs-a-gaz-et-autres-equipements/lanceurs-a-gaz-double-etage/)


Théophane Loloum, Fabien Vivodtzev, David Hébert, Baptiste Reynier, Michel Arrigoni and Julien Tierny.
