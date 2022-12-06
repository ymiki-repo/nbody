# OpenMP（target指示文）実装の概要

OpenMP（target指示文）を用いた$N$体計算コード（直接法）の実装概要の紹介

## 実装方法

* GPU上で計算させたいfor文に指示文を追加

   ```c++
   #pragma omp target teams distribute parallel for simd
   for(int32_t ii = 0; ii < num; ii++){
     ...
   }
   ```

  * simd 指示節の指定はなくても良い
  * オプション：スレッドブロックあたりのスレッド数を示唆

     ```c++
     #pragma omp target teams distribute parallel for simd thread_limit(256)
     for(int32_t ii = 0; ii < num; ii++){
       ...
     }
     ```

  * loop 指示節を使用する場合の実装（スレッド数の示唆はできない）

     ```c++
     #pragma omp target teams loop
     for(int32_t ii = 0; ii < num; ii++){
       ...
     }
     ```

* （Unified Memoryを使わない場合）データ指示文を追加
  1. GPU上のメモリ確保

     ```c++
     #pragma omp target enter data map(alloc: ptr[0:num])
     ```

  2. CPUからGPUへのデータ転送

     ```c++
     #pragma omp target update to(ptr[0:num])
     ```

  3. GPUからCPUへのデータ転送

     ```c++
     #pragma omp target update from(ptr[0:num])
     ```

  4. GPU上のメモリ解放

     ```c++
     #pragma omp target exit data map(delete: ptr[0:num])
     ```

## 実装例

| ソースコード | 実装概要 | 備考 |
| ---- | ---- | ---- |
| [cpp/openmp/0_dist/nbody_leapfrog2.cpp](/cpp/openmp/0_dist/nbody_leapfrog2.cpp) | distribute指示節を用いた実装，Unified Memoryを用いた実装，Leapfrog法 | |
| [cpp/openmp/1_dist_data/nbody_leapfrog2.cpp](/cpp/openmp/1_dist_data/nbody_leapfrog2.cpp) | distribute指示節を用いた実装，データ指示文を用いた実装，Leapfrog法 | |
| [cpp/openmp/a_loop/nbody_leapfrog2.cpp](/cpp/openmp/a_loop/nbody_leapfrog2.cpp) | loop指示節を用いた実装，Unified Memoryを用いた実装，Leapfrog法 | |
| [cpp/openmp/b_loop_data/nbody_leapfrog2.cpp](/cpp/openmp/b_loop_data/nbody_leapfrog2.cpp) | loop指示節を用いた実装，データ指示文を用いた実装，Leapfrog法 | |
| [cpp/openmp/0_dist/nbody_hermite4.cpp](/cpp/openmp/0_dist/nbody_hermite4.cpp) | distribute指示節を用いた実装，Unified Memoryを用いた実装，Hermite法 | |
| [cpp/openmp/1_dist_data/nbody_hermite4.cpp](/cpp/openmp/1_dist_data/nbody_hermite4.cpp) | distribute指示節を用いた実装，データ指示文を用いた実装，Hermite法 | 一部関数のGPU化を無効化 |
| [cpp/openmp/a_loop/nbody_hermite4.cpp](/cpp/openmp/a_loop/nbody_hermite4.cpp) | loop指示節を用いた実装，Unified Memoryを用いた実装，Hermite法 | |
| [cpp/openmp/b_loop_data/nbody_hermite4.cpp](/cpp/openmp/b_loop_data/nbody_hermite4.cpp) | loop指示節を用いた実装，データ指示文を用いた実装，Hermite法 | 一部関数のGPU化を無効化 |

* 一部関数のGPU化について
  * GPU上で動作させるとコードが正常に動作しなくなる関数があったため，暫定的にCPU上で動作させることにしている
    * CUDA版ではGPU上で正常に動作するため，実装ミスやコンパイラのバグなどが原因と考えられる
  * マクロ EXEC_SMALL_FUNC_ON_HOST を有効化（= 一部の OpenACC 指示文をコメントアウト）している
  * 小さい関数なので，実行時間への影響も小さいと考えている
  * 余分なCPU-GPU間のデータ転送が生じてしまっている

## NVIDIA HPC SDKに関する情報

### コンパイル・リンク

* 標準的な引数（コンパイル時）

  ```sh
  -mp=gpu -gpu=cc80 -Minfo=accel,opt,mp # OpenMPを使用してGPU化，cc80（NVIDIA A100）向けに最適化，GPUオフローディングや性能最適化に関するコンパイラメッセージを出力
  -mp=gpu -gpu=cc80,managed -Minfo=accel,opt # 上記に加えて，Unified Memoryを使用
  ```

* 標準的な引数（リンク時）

  ```sh
  -mp=gpu
  ```

### 実行時

* デバッグ時に便利な環境変数

  ```sh
  NVCOMPILER_ACC_NOTIFY=1 ./a.out # GPU 上でカーネルが実行される度に情報を出力する
  NVCOMPILER_ACC_NOTIFY=3 ./a.out # CPU-GPU 間のデータ転送に関する情報も出力する
  NVCOMPILER_ACC_TIME=1 ./a.out   # CPU-GPU 間のデータ転送および GPU 上での実行時間を出力する
  ```
