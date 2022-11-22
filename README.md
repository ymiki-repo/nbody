# $N$体計算コード（直接法）のGPU実装例

$N$体計算コード（直接法）を様々なGPU向けプログラミング手法で実装する

## 概要

* 各種開発環境を用いたdirect N-body codeの実装比較・性能評価
  * C++実装：CPU向けのナイーブな実装（ベースライン実装）
  * CUDA C++による実装
  * OpenACCを用いたGPUオフローディング
  * OpenMPのターゲット指示文を用いたGPUオフローディング
  * C++17の標準言語規格を用いたGPUオフローディング
* Released under the MIT license, see LICENSE.txt
* Copyright (c) 2022 Yohei MIKI

## How to compile

* Required packages:
  * CMake (>= 3.1)
  * Boost
  * HDF5
* Optional packages:
  * Julia (for visualization)
  * VisIt (for visualization)
* <details><summary>Configuration using GUI</summary>

  ```sh
  $ cmake -S. -Bbuild # source directory is the current directory, target directory is build/
  $ cd build
  $ ccmake ../        # set options using the GUI interface (CXX cannot be changed in this step)
  ```

  </details>

* <details><summary>Configuration using CUI</summary>

  ```sh
  $ cmake -S. -Bbuild [option] # source directory is the current directory, target directory is build/
  $ cd build
  ```

  </details>

* <details><summary>How to configure a fresh build tree, removing any existing cache file</summary>

  ```sh
  $ cmake --fresh -S. -Bbuild [option] # introduced in CMake 3.24
  ```

  </details>

* List of compile options (for CMake):
  * -DBENCHMARK_MODE=[ON OFF(default)] : On to perform benchmark
  * -DCALCULATE_POTENTIAL=[ON(default) OFF] : On to calculate gravitational potential
  * -DFP_L=[32(default) 64 128] : Number of bits for floating-point numbers (low-precision)
  * -DFP_M=[32 64(default) 128] : Number of bits for floating-point numbers (medium-precision)
  * -DFP_H=[64(default) 128] : Number of bits for floating-point numbers (high-precision)
  * -DHERMITE_SCHEME=[ON OFF(default)] : On to adopt 4th-order Hermite scheme instead of 2nd-order leapfrog scheme
  * -DSIMD_BITS=[256 512(default) 1024] : SIMD width in units of bit
  * -DTARGET_CPU=[depends on your C++ compiler; selecting by ccmake is encouraged] : target CPU architecture

## How to run

```sh
$ sbatch sh/slurm/run_cpu.sh [option] # run an $N$-body simulation
$ sh/slurm/check_conservation_leapfrog2.sh [option] # run a series of $N$-body simulations (check energy conservation of 2nd-order leapfrog scheme)
$ sh/slurm/check_conservation_hermite4.sh [option] # run a series of $N$-body simulations (check energy conservation of 4th-order Hermite scheme)
$ sh/slurm/check_performance_scaling.sh [option] # run a series of $N$-body simulations (evaluate time-to-solution as a function of the number of $N$-body particles)
$ sh/slurm/perform_benchmark.sh [option] # perform benchmark of direct $N$-body simulation (force calculation only)
```

* output files are dat/FILENAME_snp*.h5 and dat/FILENAME_snp*.xdmf when BENCHMARK_NODE is OFF
* log file is log/FILENAME_run.csv
* check the output figures in fig/

## How to check results

```sh
$ julia jl/plot/error.jl                                                # show time evolution of conservatives and the virial ratio
$ sbatch --export=EXEC="julia jl/plot/dot.jl" sh/slurm/plot_parallel.sh # show particles distribution by using dots
$ visit &                                                               # open dat/FILENAME_snp*.xdmf files and visualize them
```

## How to install Julia packages

```sh
$ module purge
$ module load anyenv miniconda3 # prepare Python environments
$ module load openmpi           # prepare MPI to be used
$ module load texlive           # prepare LaTeX environments
$ module load julia             # prepare Julia environments
$ julia jl/package.jl
$ julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()' # configure to use system-provided MPI
```

* <details><summary>when MPI.jl not properly configured:</summary>

  ```sh
  $ module purge
  $ module load anyenv miniconda3 # prepare Python environments
  $ module load openmpi           # prepare MPI to be used
  $ module load texlive           # prepare LaTeX environments
  $ module load julia             # prepare Julia environments
  $ julia
  > using Pkg
  > Pkg.build("MPI")
  > exit()
  ```

  </details>

## modules for Wisteria/BDEC-01 (Aquarius)

```sh
module purge # for safety
module load cmake # just for compilation
module load nvidia # NVIDIA HPC SDK
module load hdf5
```

## Profiling

### NVIDIA GPU向け

* とりあえず nsys を動かしてみる

  ```sh
  $ nsys profile --stats=true ./a.out # --stats=true をつけておくと，標準エラー出力にも結果（の概要）が出てくる
  $ nsys-ui & # これはGUIツールなので手元で開く方が良い
  ```

  1. nsys-ui 上で，nsys によって生成された report?.nsys-rep を開く
  2. タイムライン左側の"CUDA HW..."というところを右クリックして"Show in Events View"をクリック
  3. 下側のウィンドウで注目している関数名をクリックすると，当該カーネルの情報が出てくる（スレッド数などの基礎的情報．キャッシュヒット率などの詳細な情報ではない）
