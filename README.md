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

1. Download the reference dataset : [90° impact (1) (344 MB)](https://zenodo.org/records/19248484/files/animation_s1.vti) and move it to the 'example' directory.

2. Go to the `example` directory
 
3. Next, from there, enter the following command:
```
paraview example.pvsm
```

This reproduces the images from the Figure 1 of the manuscript *DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging*.

## Downloading the database
We contribute our database of hypervelocity impact acquisitions, i.e., 7 other two-dimensional time-varying scalar fields (about 2 gigabytes), counting between 256 and 1050 frames, capturing different dynamic fragmentation modalities (projectile-based and laser-based).
- [90° impact (2) (463 MB)](https://zenodo.org/records/19250540/files/90_HE0905.vti)
- [90° impact (3) (546 MB)](https://zenodo.org/records/19250595/files/90_HE09013.vti)
- [90° impact (4) (558 MB)](https://zenodo.org/records/19250495/files/90_HE0904.vti)
- [90° impact (5) (67 MB)](https://zenodo.org/records/19249889/files/mica18.vti)
- [45° impact (1) (216 MB)](https://zenodo.org/records/19250442/files/45_HE0911.vti)
- [45° impact (2) (224 MB)](https://zenodo.org/records/19250008/files/45_HE908.vti)
- [Laser impact (204 MB)](https://zenodo.org/records/19250725/files/Belenos_laser.vti)

## Reference

DebrisTracer: Reliable Tracking in Hypervelocity Impact Fast Imaging

Théophane Loloum, Fabien Vivodtzev, David Hébert, Baptiste Reynier, Michel Arrigoni and Julien Tierny.
