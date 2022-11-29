# $N$体計算コード（直接法）のGPU実装例

$N$体計算コード（直接法）を様々なGPU向けプログラミング手法で実装する

## 概要

* 各種開発環境を用いたdirect $N$-body codeの実装比較・性能評価
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
  * <details><summary>How to load modules on Wisteria/BDEC-01 (Aquarius): NVIDIA HPC SDK</summary>

    ```sh
    module purge       # for safety
    module load cmake  # CMake: just for compilation
    module load nvidia # NVIDIA HPC SDK
    module load hdf5   # HDF5
    ```

  </details>

  * <details><summary>How to load modules on Wisteria/BDEC-01 (Aquarius): CUDA</summary>

    ```sh
    module purge      # for safety
    module load cmake # CMake: just for compilation
    module load cuda  # CUDA
    module load gcc   # GCC: required to load hdf5
    module load hdf5  # HDF5
    ```

  </details>

* Optional packages:
  * Julia (for visualization)
  * VisIt (for visualization)
* <details><summary>Configuration using GUI</summary>

  ```sh
  cmake -S. -Bbuild # source directory is the current directory, target directory is build/
  cd build
  ccmake ../        # set options using the GUI interface (CXX cannot be changed in this step)
  ```

  </details>

* <details><summary>Configuration using CUI</summary>

  ```sh
  cmake -S. -Bbuild [option] # source directory is the current directory, target directory is build/
  cd build
  ```

  </details>

* <details><summary>How to configure a fresh build tree, removing any existing cache file</summary>

  ```sh
  cmake --fresh -S. -Bbuild [option] # introduced in CMake 3.24
  ```

  </details>

* List of compile options (for CMake):
  * -DBENCHMARK_MODE=[ON OFF(default)] : On to perform benchmark
  * -DCALCULATE_POTENTIAL=[ON(default) OFF] : On to calculate gravitational potential
  * -DFP_L=[32(default) 64 128] : Number of bits for floating-point numbers (low-precision)
  * -DFP_M=[32 64(default) 128] : Number of bits for floating-point numbers (medium-precision)
  * -DFP_H=[64(default) 128] : Number of bits for floating-point numbers (high-precision)
  <!-- 実装中* -DHERMITE_SCHEME=[ON OFF(default)] : On to adopt 4th-order Hermite scheme instead of 2nd-order leapfrog scheme -->
  * -DSIMD_BITS=[256 512(default) 1024] : SIMD width in units of bit
  * -DTARGET_CPU=[depends on your C++ compiler; selecting by ccmake is encouraged] : target CPU architecture

* Compilation
```sh
ninja # if ninja-build is installed
make  # if ninja-build is missing
```

## How to run

* <details><summary>Wisteria/BDEC-01 (Fujitsu TCS)</summary>

  ```sh
  pjsub sh/wisteria/run_nvidia.sh # run an $N$-body simulation in default configuration, base compiler is nvhpc
  pjsub sh/wisteria/run_cuda.sh # run an $N$-body simulation in default configuration, base compiler is cuda
  pjsub -x EXEC=bin/acc_managed,OPTION="--num=16384 --file=acc" sh/wisteria/run_nvidia.sh # run an $N$-body simulation with option (binary is bin/acc_managed, $N = 16384$, FILENAME is acc), base compiler is nvidia
  pjsub -x EXEC=bin/cuda_memcpy_base,OPTION="--num=16384 --file=cuda_memcpy" sh/wisteria/run_cuda.sh # run an $N$-body simulation with option (binary is bin/cuda_memcpy_base, $N = 16384$, FILENAME is cuda_memcpy), base compiler is cuda
  ```

  </details>

* <details><summary>Slurm</summary>

  ```sh
  sbatch sh/slurm/run.sh [option] # run an $N$-body simulation
  sh/slurm/check_conservation_leapfrog2.sh [option] # run a series of $N$-body simulations (check energy conservation of 2nd-order leapfrog scheme)
  sh/slurm/check_conservation_hermite4.sh [option] # run a series of $N$-body simulations (check energy conservation of 4th-order Hermite scheme)
  sh/slurm/check_performance_scaling.sh [option] # run a series of $N$-body simulations (evaluate time-to-solution as a function of the number of $N$-body particles)
  sh/slurm/perform_benchmark.sh [option] # perform benchmark of direct $N$-body simulation (force calculation only)
  ```

  </details>

* output files are dat/FILENAME_snp*.h5 and dat/FILENAME_snp*.xdmf when BENCHMARK_NODE is OFF
* log file is log/FILENAME_run.csv

<!-- MEMO: Julia script does not work on Wisteria/BDEC-01 (perhaps, the reason is version of installed TexLive) -->
<!-- ## 可視化のための事前準備（Python および Julia を使用する場合）

1. Matplotlib環境の構築
   * <details><summary>Wisteria/BDEC-01 (Aquarius) 向けの環境構築</summary>

     ```sh
     mkdir -p /work/{YOUR_GROUP}/$USER/opt/$(uname -m) # 以下，{YOUR_GROUP} は全てご自分の所属グループに置き換えてください
     cp -r modules /work/{YOUR_GROUP}/$USER/opt/
     # /work/{YOUR_GROUP}/$USER/opt/anyenv 14行目の gz00 をご自分の所属グループに編集してください（必須）
     cd /work/{YOUR_GROUP}/$USER/opt/$(uname -m) # Aquarius（x86_64環境）用の環境と，Odyssey（aarch64環境）用の環境を分離して構築できるようにするための工夫
     git clone https://github.com/anyenv/anyenv
     module use /work/{YOUR_GROUP}/$USER/opt/modules
     module load anyenv
     anyenv install --init # y/N を聞かれるので，y とする
     git clone https://github.com/znz/anyenv-update.git $(anyenv root)/plugins/anyenv-update
     anyenv update # このコマンドによって，後で導入する pyenv なども update されるようになる
     anyenv install pyenv
     pyenv install -l | grep miniconda3 # インストールできるバージョンを確認（miniforge3でも良い）
     pyenv install miniconda3-4.7.12
     pyenv rehash
     pyenv global miniconda3-4.7.12
     pyenv versions
     cd /work/{YOUR_GROUP}/$USER/opt/modules
     cd miniconda3 # miniforge3 をインストールした場合にはフォルダ名を miniforge3 に mv した上で cd してください
     ln -s .generic 4.7.12 # これは v4.7.12 をインストールした場合です
     touch /work/{YOUR_GROUP}/$USER/.condarc
     mkdir /work/{YOUR_GROUP}/$USER/.conda
     mv ~/.condarc ~/.condarc.bak # もしあれば
     mv ~/.conda ~/.conda.bak # もしあれば
     ln -s /{YOUR_GROUP}/$USER/.conda* ~/
     conda config --env --remove channels defaults
     conda config --env --add channels conda-forge
     # お好みのエディタで /work/{YOUR_GROUP}/$USER/.config/$(uname -m)/.condarc を開き，下記2行を追記（オプション，容量を節約したい場合）
     # allow_softlinks: true
     # always_softlink: true
     conda update --all
     conda install matplotlib
     ```

     </details>

1. Julia環境の構築
   * <details><summary>Wisteria/BDEC-01上での環境構築</summary>

     ```sh
     # 1. sh/wisteria/setup_julia.sh 15-18行目の Python 環境の設定をご自分の環境に合わせて編集してください（上記設定の通りにPython環境を構築した場合にはこの手順は不要）
     pjsub --vset PROJ=gz00 -x PROJ=gz00 sh/wisteria/setup_julia.sh # gz00 をご自分の所属グループに編集してください（必須，2ヶ所あります）
     ```

     </details>
   * <details><summary>お手元の環境などでの構築方法</summary>

     ```sh
     module load miniconda
     module load anyenv miniconda3 # prepare Python environments
     module load openmpi           # prepare MPI to be used
     module load texlive           # prepare LaTeX environments
     module load julia             # prepare Julia environments
     julia jl/package.jl
     julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()' # configure to use system-provided MPI
     ```

     </details>

## How to check results

 * <details><summary>Wisteria/BDEC-01上での実行方法</summary>

   ```sh
   pjsub --vset PROJ=gz00 -x PROJ=gz00,OPTION="--target=FILENAME" sh/wisteria/plot_error.sh # エネルギー保存などの時間進化を描画，gz00 をご自分の所属グループに編集してください（必須，2ヶ所あります）．FILENAMEは$N$体計算実行時に--file=として指定したものです
   pjsub --vset PROJ=gz00 -x PROJ=gz00,OPTION="--target=FILENAME" sh/wisteria/plot_dot.sh   # 粒子分布の時間進化を描画，gz00 をご自分の所属グループに編集してください（必須，2ヶ所あります）．FILENAMEは$N$体計算実行時に--file=として指定したものです
   ```

   </details>

 * <details><summary>Slurm環境などでの実行方法</summary>

   ```sh
   julia jl/plot/error.jl                                                # show time evolution of conservatives and the virial ratio
   sbatch --export=EXEC="julia jl/plot/dot.jl" sh/slurm/plot_parallel.sh # show particles distribution by using dots
   visit &                                                               # open dat/FILENAME_snp*.xdmf files and visualize them
   ```

   </details>

* check the output figures in fig/ -->

## Profiling

### NVIDIA GPU向け

* とりあえず nsys を動かしてみる

  ```sh
  nsys profile --stats=true ./a.out # --stats=true をつけておくと，標準エラー出力にも結果（の概要）が出てくる
  nsys-ui & # これはGUIツールなので手元で開く方が良い
  ```

  1. nsys-ui 上で，nsys によって生成された report?.nsys-rep を開く
  2. タイムライン左側の"CUDA HW..."というところを右クリックして"Show in Events View"をクリック
  3. 下側のウィンドウで注目している関数名をクリックすると，当該カーネルの情報が出てくる（スレッド数などの基礎的情報．キャッシュヒット率などの詳細な情報ではない）
