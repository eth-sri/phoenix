# Phoenix <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

Phoenix is a system for privacy-preserving neural network inference with reliability guarantees, described in our [research paper][jovanovic2022phoenix]:

> Nikola Jovanović, Marc Fischer, Samuel Steffen, and Martin Vechev. 2022. _Private and Reliable Neural Network Inference._ In Proceedings of CCS ’22.

[jovanovic2022phoenix]: https://www.sri.inf.ethz.ch/publications/jovanovic2022phoenix

For a brief overview, check out our **[blogpost](https://www.sri.inf.ethz.ch/blog/phoenix)**.

### Installation

First, set up a conda environment with pytorch, needed to run the python code that prepares the data and models used by Phoenix:

```
conda create -n phoenix_env python=3.7
conda activate phoenix_env
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
```

Phoenix itself is implemented in C++. To run it, add `cmake` and `hcephes` packages to your conda environment:
```
conda install cmake
conda install -c conda-forge hcephes
```

Next, set up the [Microsoft SEAL 3.6.6](https://github.com/microsoft/SEAL) FHE library. First, clone the repository:

```
cd ~
mkdir libs
git clone https://github.com/microsoft/SEAL.git
cd SEAL
git checkout v3.6.6
```

Next, we have to manually fix a broken dependency setup, and add a definition to the header file:
```
sed -i 's/1.1.0/1.2.3/g' CMakeLists.txt 
sed -i 's/2dc1db/0858760dc957280e8eb8953af4b4b83879d7b8a4/g' cmake/ExternalIntelHEXL.cmake
echo -e "\n#define SEAL_COEFF_MOD_COUNT_MAX 128\n" >> native/src/seal/c/defines.h
```

Finally we can install SEAL:
```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=~/libs -DSEAL_USE_INTEL_HEXL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```

Phoenix further uses [rapidcsv](https://github.com/d99kris/rapidcsv) and [jsoncpp](http://jsoncpp.sourceforge.net/) libraries, which are directly included in our source. We can proceed to build Phoenix, as well as the variant that uses MockSEAL, as described in the paper. From the project root run:

```
cd phoenix
cmake .
make phoenix
make mockphoenix
mkdir out
```

### Running Phoenix

It is necessary to first prepare the corresponding dataset and trained models to use for reliable inference. The python code for this is given in `phoenix/_torch`. For example, to generate data for the MNIST dataset, run the following:

```
cd phoenix 
mkdir ../data
cd phoenix/_torch
python3 train.py save_data mnist
```

Use a similar command for CIFAR; for Adult, refer to `train_adult.py`. We provide all models used in our evaluation in `weights/`, e.g., `weights/mnist_mlp2_0.5.csv` represents the MLP2 network trained with noise 0.5 on the MNIST dataset, using the command line `python3 train.py train mnist 0.5`.

To finally run Phoenix, see the `scripts/` directory, which provides a set of example commands used to produce Table 2 in our paper. We rely on config files (see `configs/`), which specify the dataset, network, and all parameters needed to perform reliable inference. As an example, to run Phoenix with SEAL on example index 1 from the MNIST dataset use:
```
./phoenix configs/mnist_mlp2.json 1
```
This will load the model `weights/mnist_mlp2_0.5.csv` and data from `data_mnist_test_10k.csv` and run robustness certification with randomized smoothing according to parameters in the config file (both client and server side of the protocol), reporting the results on standard output, and appending the result string to `out_path` (`out/mnist_mlp2.txt` in this case). To obtain the numbers presented in Table 2, we used the `process_results.py` script to aggregate the results of a large number of runs.

### Citation

If you use Phoenix please cite the following.

```
@inproceedings{jovanovic2022phoenix,
    author = {Jovanović, Nikola and Fischer, Marc and Steffen, Samuel Vechev, Martin},
    title = {Private and Reliable Neural Network Inference},
    year = {2022},
    publisher = {Association for Computing Machinery},
    booktitle = {Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security},
    location = {Los Angeles, U.S.A.},
    series = {CCS ’22}
}
```
