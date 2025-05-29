This LAMMPS pair style enables the use of a structure-preserving machine-learned framework for coarse-grained simulations.

## Pre-requisites
LibTorch == 2.1.2

Libtorch can be downloaded here: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcpu.zip (Preferred method)

MKL Package
```
pip install mkl-include
```
It can be installed using conda as well.

Other packages/modules loaded/used:
gcc-13.3 or gcc-9
openmpi-2.1.0

## Building LAMMPS
Currently this potential has only been tested for the latest LAMMPS version (29Aug2024).
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
cmake -DPKG_RIGID=yes -DPKG_SPH=yes -DPKG_EXTRA-DUMP=yes  -DBUILD_MPI=yes -DPKG_EXTRA-COMPUTE=yes -DBUILD_OMP=yes ../cmake/ -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build .
```

If you have PyTorch installed:

```
bash
cd lammps
mkdir build
cd build
cmake -DPKG_RIGID=yes -DPKG_SPH=yes -DPKG_EXTRA-DUMP=yes -DPKG_EXTRA-COMPUTE=yes -DBUILD_MPI=yes  -DBUILD_OMP=yes ../cmake/ -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
cmake --build .
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
python main_compile.py --lammps /path/to/lammps/executable --lammps_setup /path/to/lammps/setup/input/file --lammps_sim_dir /path/to/lammps/sim/ --lmp_setup_data pre_entropy.data --lmp_entropy_data ig_entropy.data --metadata /path/to/metadata/ --params /path/to/params/file
```
`main_compile.py` will take an existing `params.pt` that the user has already trained and compile it into three seperate JIT PyTorch models that can be used within LAMMPS. It produces 
'model_jit_s.pt', `model_jit_w.pt`, and `model_jit.pt`, which are used to calculate the entropies, volumes, and the forces + dS, respectively. It will also generate a LAMMPS simulation directory containing a LAMMPS data file with the initial per-particle entropies calculated based on the user's LAMMPS setup file, three JIT models, `sdpd_exec.in`, which is a LAMMPS input script with the timestep and cutoff defined during the training of the model. The arguments for `main_compile.py` are as follows:

|     Argument                  |             Description                                                                                | Options                                               |
|-------------------------------| -------------------------------------------------------------------------------------------------------|------------------------------------------------------ |
| `--lammps`                    | Location of LAMMPS binary                                                                              | /path/to/lammps/binary                                |
| `--lammps_setup`              | LAMMPS file generates initial parameters                                                               | /path/to/lammps/input                                 | 
| `--lammps_sim_dir`            | LAMMPS folder to perform simulations with sdpd/ml pair style; will be made unless it exists            | /path/to/directory/                                   |
| `--lmp_setup_data`            | Name of LAMMPS data file that is produced by the `--lammps_setup` input file                           | name of LAMMPS data file                              |
| `--lmp_entropy_data`          | Name of LAMMPS data file after the entropy calculation is performed                                    | name of LAMMPS data file                              |
| `--lmp_custom_input`          | Location of custom LAMMPS input file (ddpd_exec.in will not be generated)                              | /path/to/custom/input/file                            |
| `--metadata`                  | Location of the metadata of the training parameters                                                    | /path/to/metadata                                     |
| `--params`                    | Location of trained PyTorch model that will be JIT compiled                                            | /path/to/params.pt                                    |    


As an example, the following command can be used to generate the `sp11_sim` directory (note that we are compiling the pre-trained star polymer 11 dataset that comes with this package):
```
python main_compile.py --lammps /path/to/lmp/executable --lammps_setup lammps_setup_examples/lammps_star_polymer11/sp11_setup.in --lammps_sim_dir sp11_sim --lmp_setup_data pre_entropy.data --lmp_entropy_data sp11_entropy.data --metadata trained_models/star_polymer_11/args.json --params training_data/star_polymer_11/params.pt
```

### Guidelines to remember:

Within your `--lmp_setup_data`, the user should generate initial configuration that is representative of your model's training data. For example, the initial velocities created should reflect that of the training data.

Make sure to use the SPH atom style, but please know that the column formatting in the LAMMPS data files is altered after patching with LAMMPS with `sdpd/ml` pair style because of the additional per-particle entropy. This is the current column formatting: 
```
atom-id atom-type rho esph cv entropy x y z ix iy iz
```
### Dependencies:
```
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install MDAnalysis
```
*Note that the PyTorch(2.1.2) and LibTorch(2.5.0) versions are different.

## LAMMPS Input File
```
pair_style	ddpd/ml <random_seed>
pair_coeff	* * <cutoff> model_jit_W.pt model_jit.pt
```
If you use `main_compile.py`, then the `<cutoff>` is already defined by your training metadata and `<random_seed>` is generated using `numpy`.

## Full tutorial 

To see a full walkthrough of training a model, compiling LAMMPS with `sdpd/ml` pairstyle, running LAMMPS with a trained model, check out the following link:
https://colab.research.google.com/drive/1ZKeimm3Eeo_fF9WrPkzcCnFEj55xhkce?usp=sharing
