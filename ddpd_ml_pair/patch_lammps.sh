#!/bin/bash
# patch_lammps.sh [-e] /path/to/lammps/
do_e_mode=false
while getopts "hek" option; do
   case $option in
      e)
         do_e_mode=true;;
      h) # display Help
         echo "patch_lammps.sh [-e] /path/to/lammps/"
         exit;;
   esac
done
# https://stackoverflow.com/a/9472919
shift $(($OPTIND - 1))
lammps_dir=$1
if [ "$lammps_dir" = "" ];
then
    echo "lammps_dir must be provided"
    exit 1
fi
if [ ! -d "$lammps_dir" ]
then
    echo "$lammps_dir doesn't exist"
    exit 1
fi
if [ ! -d "$lammps_dir/cmake" ]
then
    echo "$lammps_dir doesn't look like a LAMMPS source directory"
    exit 1
fi
# Check and produce nice message
if [ ! -f pair_ddpd_ml.cpp ]; then
    echo "Please run `patch_lammps.sh` from the ddpd_ml_pair directory."
    exit 1
fi
# Check for double-patch
if grep -q "find_package(Torch REQUIRED)" $lammps_dir/cmake/CMakeLists.txt ; then
    echo "This LAMMPS installation _seems_ to already have been patched; please check it!"
    # exit 1
fi
if [ "$do_e_mode" = true ]
then 
    echo "Removing atom.*, set.*,SPH/atom_vec_sph.* files"
    rm $lammps_dir/src/atom.*	
    rm $lammps_dir/src/SPH/atom_vec_sph.* 
    rm $lammps_dir/src/set.*
    echo "Making source symlinks (-e)..."
    ln -s `realpath -s atom.h` $lammps_dir/src/atom.h
    ln -s `realpath -s atom.cpp` $lammps_dir/src/atom.cpp
    ln -s `realpath -s set.h` $lammps_dir/src/set.h
    ln -s `realpath -s set.cpp` $lammps_dir/src/set.cpp    
    ln -s `realpath -s atom_vec_sph.h` $lammps_dir/src/SPH/atom_vec_sph.h
    ln -s `realpath -s atom_vec_sph.cpp` $lammps_dir/src/SPH/atom_vec_sph.cpp
    for file in *ddpd*; do
        ln -s `realpath -s $file` $lammps_dir/src/SPH/$file
    done
else
    echo "Copying files..."
    cp atom.h $lammps_dir/src/atom.h
    cp atom.cpp $lammps_dir/src/atom.cpp
    cp set.h $lammps_dir/src/set.h
    cp set.cpp $lammps_dir/src/set.cpp
    cp atom_vec_sph.h $lammps_dir/src/SPH/atom_vec_sph.h
    cp atom_vec_sph.cpp $lammps_dir/src/SPH/atom_vec_sph.cpp
    for file in *ddpd*; do
        cp $file $lammps_dir/src/SPH/$file
    done
fi
echo "Updating CMakeLists.txt..."
sed -i "s/set(CMAKE_CXX_STANDARD 11)/set(CMAKE_CXX_STANDARD 17)/" $lammps_dir/cmake/CMakeLists.txt
# Add libtorch
cat >> $lammps_dir/cmake/CMakeLists.txt << "EOF2"
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PUBLIC "${TORCH_LIBRARIES}")
EOF2
echo "Done!"

