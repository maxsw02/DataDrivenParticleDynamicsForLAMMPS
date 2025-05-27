This LAMMPS pair style allows you to use structure-preserving machine-learned framework for DPD simulations. 

## Pre-requisites
PyTorch or LibTorch == 2.1.2

Libtorch an be downloaded here: https://dev-discuss.pytorch.org/t/pytorch-release-2-1-2-final-rc-is-available/1708 (Preferred method)

Pytorch 2.1.2 can be installed using pip/conda: https://pytorch.org/get-started/previous-versions/

MKL Package
``` 
conda install conda-forge::mkl
```
Can be installed using pip as well.

Other packages/modules loaded/used:
gcc-9
openmpi-2.1.0

## Building LAMMPS
Currently this potential has only been tested for the latest LAMMPS version (19Nov2024).
Could work for older versions, but not tested.
Feel free to download the latest version of LAMMPS from their github page: https://github.com/lammps/lammps.
Or use the provided patched version of LAMMPS in this repository/directory.

### Patching LAMMPS

#### Via patch script

From this directory run: 
```bash
./patch_lammps_sdpd.sh /path/to/lammps/
```
This step is to ensure that LAMMPS C++ code and Pytorch  C++ codes are recognizing one another.

#### Manually
First copy the source files of the pair style:
```bash
cp /path/to/sdpd_ml_pair/atom_vec_sph.* /path/to/lammps/src/SPH
cp /path/to/sdpd_ml_pair/fix_sdpd.* /path/to/lammps/src/USER-SPH
cp /path/to/sdpd_ml_pair/pair_sdpd_ml.* /path/to/lammps/src/SPH

cp /path/to/sdpd_ml_pair/set.* /path/to/lammps/src/
cp /path/to/sdpd_ml_pair/atom.* /path/to/lammps/src/

```
Then make the following modifications to `lammps/cmake/CMakeLists.txt`:
- Append the following lines:
```cmake
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
```
### Compiling LAMMPS
If you don't have PyTorch installed, you need to download LibTorch from the [PyTorch download page](https://pytorch.org/get-started/locally/). Unzip the downloaded file, then configure LAMMPS:

```
bash
cd lammps
mkdir build
cd build
cmake -DPKG_RIGID=yes -DPKG_SPH=yes -DPKG_EXTRA_DUMP=yes  -DBUILD_MPI=yes  -DBUILD_OMP=yes ../cmake/ -DCMAKE_PREFIX_PATH=/path/to/libtorch 
```

If you have PyTorch installed:

```
bash
cd lammps
mkdir build
cd build
cmake -DPKG_RIGID=yes -DPKG_SPH=yes -DPKG_EXTRA_DUMP=yes  -DBUILD_MPI=yes  -DBUILD_OMP=yes ../cmake/ -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` 
```

CMake will look for MKL. You may have to explicitly provide the path for your MKL installation (e.g. `-DMKL_INCLUDE_DIR=/usr/include/`).

**MKL:** If `MKL_INCLUDE_DIR` is not found and you are using a Python environment, a simple solution is to run `conda install mkl-include` or `pip install mkl-include` and append:
```
-DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
```
to the `cmake` command if using a `conda` environment, or
```
-DMKL_INCLUDE_DIR=`python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"`
```
if using plain Python and `pip`.

# Usage

## Before LAMMPS

### Generating LAMMPS Input Files

```
python generate_input_files.py
```

This step uses model_jit_s.pt to calculate the initial entropies of each particle based on its initial positions and velocities before running LAMMPS. It also generates a input LAMMPS data file, `atoms_mod.data`, with the corresponding initial entropies, and lammps_input file, `sdpd_exec.in`. If you changed thenames of pytorch models, then change the appropriate lines in the sdpd_exec.in file.

Within `sdpd_exec.in` produced by `generate_input_files.py` , the initial density, distribution of the initial velocities, interaction cutoff, and timestep should match that of the training data. The current `sdpd_exec.in` file is meant for the star_polymer_11 dataset.

## LAMMPS
```
pair_style	sdpd/ml <random_seed>
pair_coeff	* * <cutoff> <torch_modelW>.pt <torch_model>.pt
```



