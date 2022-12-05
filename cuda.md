# CUDA C++を用いた実装の概要

CUDA C++を用いた$N$体計算コード（直接法）の実装概要の紹介

## 実装方法

* GPU上で計算させたい関数を\_\_global\_\_関数へと書き換える
  * 関数定義の先頭に \_\_global\_\_ をつける

   ```c++
   __global__ void func_device(double *__restrict arg0, int32_t *__restrict arg1, ...)
   ```

  * 一番外側のfor文を削除し，代わりに（自動的に設定される）スレッドIDを用いる
    * for文をif文に置き換えるよう誘導されることが多いが，ここでは不要なif文を削除するための工夫を施すこととする
      * 領域外参照が起こらないように，メモリ確保時に「スレッドブロックあたりのスレッド数」の定数倍の要素数を確保する
      * （余分な粒子については質量を$0$とするなどして，計算結果に影響を与えないよう注意する）

   ```c++
   // for(int32_t ii = 0; ii < num; ii++){
   const int32_t ii = blockIdx.x * blockDim.x + threadIdx.x;
   ```


  * GPU上の関数から呼び出す関数については，関数定義の先頭に\_\_device\_\_をつける

     ```c++
     __device__ void sub_func_device(double *__restrict arg0, int32_t *__restrict arg1, ...)
     ```

  * CPUからGPU上の\_\_global\_\_関数を起動する
    * NTHREADS はスレッドブロックあたりのスレッド数であり，性能への影響が大きいパラメータである

   ```c++
   func_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS>>>(arg0, arg1, ...);
   ```

  * スレッドブロックのブロック数についても指定する必要があるが，下記のマクロ関数を利用するのが便利である

   ```c++
   constexpr int32_t BLOCKSIZE(const int32_t num, const int32_t thread) { return (1 + ((num - 1) / thread)); }
   ```

* GPU上のメモリに関する実装
  * Unified Memory を使用する場合
    1. メモリを確保する

     ```c++
     cudaMallocManaged((void **)ptr, size_of_array_in_byte);
     ```

    2. メモリを解放する

     ```c++
     cudaFree(ptr);
     ```

  * cudaMemcpy を用いて明示的にCPU-GPU間のデータ転送を記述する場合
    1. メモリを確保する

     ```c++
     cudaMalloc((void **)ptr_dev, size_of_array_in_byte);     // GPU上にメモリを確保
     cudaMallocHost((void **)ptr_hst, size_of_array_in_byte); // CPU上にpinnedメモリを確保（pinnedしておいた方がCPU-GPU間の転送が（特に容量の大きいところで）速い）
     ```

    2. CPU-GPU間のデータ転送を行う

     ```c++
     cudaMemcpy(dst_dev, src_hst, size_of_array_in_byte, cudaMemcpyHostToDevice); // CPU上の*src_hstからGPU上の*dst_devへとデータ転送
     cudaMemcpy(dst_hst, src_dev, size_of_array_in_byte, cudaMemcpyDeviceToHost); // GPU上の*src_devからCPU上の*dst_hstへとデータ転送
     ```

    3. メモリを解放する

     ```c++
     cudaFree(ptr_dev);
     cudaFreeHost(ptr_hst);
     ```

* （必要に応じて）GPU上での処理の完了待ちを行う
  * cudaDeviceSynchronize() がいらない場合
    * デフォルトストリームのみを用いて計算し，CPU-GPU間のデータ転送にcudaMemcpyのみを用いている場合
  * cudaDeviceSynchronize() がいる場合
    * 複数のCUDAストリームを使う，AsynchronousなCUDA関数を使用した場合など
    * Unified Memory を用いた際などに，CPUから読み出したデータが正しくなかった場合
    * 性能測定時など，GPU上の関数の実行状態を正確に把握しておく必要がある場合

 ```c++
 cudaDeviceSynchronize();
 ```
## 実装例

| ソースコード | 実装概要 |
| ---- | ---- |
| [cpp/cuda/00_unified_base/nbody_leapfrog2.cu](/cpp/cuda/00_unified_base/nbody_leapfrog2.cu) | Unified Memoryを用いた実装，Leapfrog法 |
| [cpp/cuda/00_unified_base/nbody_hermite4.cu](/cpp/cuda/00_unified_base/nbody_hermite4.cu) | Unified Memoryを用いた実装，Hermite法 |
| [cpp/cuda/01_unified_rsqrt/nbody_leapfrog2.cu](/cpp/cuda/01_unified_rsqrt/nbody_leapfrog2.cu) | Unified Memoryを用いた実装，rsqrt()関数を利用，Leapfrog法 |
| [cpp/cuda/01_unified_rsqrt/nbody_hermite4.cu](/cpp/cuda/01_unified_rsqrt/nbody_hermite4.cu) | Unified Memoryを用いた実装，rsqrt()関数を利用，Hermite法 |
| [cpp/cuda/02_unified_shmem/nbody_leapfrog2.cu](/cpp/cuda/02_unified_shmem/nbody_leapfrog2.cu) | Unified Memoryを用いた実装，rsqrt()関数を利用，シェアードメモリを利用，Leapfrog法 |
| [cpp/cuda/02_unified_shmem/nbody_hermite4.cu](/cpp/cuda/02_unified_shmem/nbody_hermite4.cu) | Unified Memoryを用いた実装，rsqrt()関数を利用，シェアードメモリを利用，Hermite法 |
| [cpp/cuda/10_memcpy_base/nbody_leapfrog2.cu](/cpp/cuda/10_memcpy_base/nbody_leapfrog2.cu) | cudaMemcpy()を用いた実装，Leapfrog法 |
| [cpp/cuda/10_memcpy_base/nbody_hermite4.cu](/cpp/cuda/10_memcpy_base/nbody_hermite4.cu) | cudaMemcpy()を用いた実装，Hermite法 |
| [cpp/cuda/11_memcpy_rsqrt/nbody_leapfrog2.cu](/cpp/cuda/11_memcpy_rsqrt/nbody_leapfrog2.cu) | cudaMemcpy()を用いた実装，rsqrt()関数を利用，Leapfrog法 |
| [cpp/cuda/11_memcpy_rsqrt/nbody_hermite4.cu](/cpp/cuda/11_memcpy_rsqrt/nbody_hermite4.cu) | cudaMemcpy()を用いた実装，rsqrt()関数を利用，Hermite法 |
| [cpp/cuda/12_memcpy_shmem/nbody_leapfrog2.cu](/cpp/cuda/12_memcpy_shmem/nbody_leapfrog2.cu) | cudaMemcpy()を用いた実装，rsqrt()関数を利用，シェアードメモリを利用，Leapfrog法 |
| [cpp/cuda/12_memcpy_shmem/nbody_hermite4.cu](/cpp/cuda/12_memcpy_shmem/nbody_hermite4.cu) | cudaMemcpy()を用いた実装，rsqrt()関数を利用，シェアードメモリを利用，Hermite法 |

## 性能最適化

* シェアードメモリの活用
  * オンチップの高速（小容量）なメモリであり，スレッドブロック内の全スレッドからアクセスできる
    * シェアードメモリ上のデータの一貫性はユーザが保証しなければならないので，適宜 \_\_syncthreads() などを発行する
  * static allocation（\_\_global\_\_関数などの中で確保する）の場合（[実装例](/cpp/cuda/12_memcpy_shmem/nbody_leapfrog2.cu)）
    * こちらの方法の方が楽なので，基本的には static allocation がお勧め
    * この方法で確保できるブロックあたりのシェアードメモリ容量の上限は48 KB
      * static allocation できない際には，後述の dynamic allocation を使う

   ```c++
   __shared__ double shmem[NTHREADS];
   ```

  * dynamic allocation（\_\_global\_\_関数を起動する際に動的に確保する）の場合（[実装例](/cpp/cuda/12_memcpy_shmem/nbody_hermite2.cu)）

   ```c++
   extern __shared__ double shmem[]; // __global__関数の外側に書いておく

   double *ptr0_shmem = (double *)shmem; // __global__関数の中の記述
   double *ptr1_shmem = (double *)&ptr0_shmem[NTHREADS]; // __global__関数の中の記述

   func_device<<<BLOCKSIZE(num, NTHREADS), NTHREADS, allocated_size_in_byte>>>(arg0, arg1, ...); // __global__関数呼び出し時に引数（シェアードメモリの容量）を追加
   ```

* 高速な演算命令の活用
  * $N$体計算の場合には，逆数平方根の高速化が最重要
  * NVIDIA GPU上では，rsqrtf() を使うことで高速化される
    * 単精度よりも少しだけ精度が悪いが，実用上問題ない
    * 倍精度演算の場合には，rsqrtf() の結果をシードとしてNewton--Raphson法を用いる方が，1.0 / sqrt() とする（平方根を計算した後に除算が処理される）よりも高速である
    * [実装例](/cpp/cuda/11_memcpy_rsqrt/nbody_leapfrog2.cu)

## NVIDIA CUDA SDKに関する情報

### コンパイル・リンク

* 標準的な引数（コンパイル時）

  ```sh
  -gencode arch=compute_80,code=sm_80 -Xptxas -v,-warn-spills,-warn-lmem-usage -lineinfo # cc80（NVIDIA A100）向けに最適化，注視しておきたい情報を出力
  ```
