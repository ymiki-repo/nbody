# OpenACC 実装の概要

OpenACC を用いた$N$体計算コード（直接法）の実装概要の紹介

## 実装方法

* GPU上で計算させたいfor文に指示文を追加（`kernels` で GPU化されない場合には `parallel` の使用を検討）

   ```c++
   #pragma acc kernels
   #pragma acc loop independent
   for(int32_t ii = 0; ii < num; ii++){
     ...
   }
   ```

  * オプション：スレッドブロックあたりのスレッド数を示唆（`kernels` で GPU化されない場合には `parallel` の使用を検討）

     ```c++
     #pragma acc kernels vector_length(256)
     #pragma acc loop independent
     for(int32_t ii = 0; ii < num; ii++){
       ...
     }
     ```

  * スレッドブロックあたりのスレッド数は，下記の記法でも示唆できる

     ```c++
     #pragma acc kernels
     #pragma acc loop independent vector(256)
     for(int32_t ii = 0; ii < num; ii++){
       ...
     }
     ```

* （Unified Memoryを使わない場合）データ指示文を追加
  1. GPU上のメモリ確保

     ```c++
     #pragma acc enter data create(ptr[0:num])
     ```

  2. CPUからGPUへのデータ転送

     ```c++
     #pragma acc update device(ptr[0:num])
     ```

  3. GPUからCPUへのデータ転送

     ```c++
     #pragma acc update host(ptr[0:num])
     ```

  4. GPU上のメモリ解放

     ```c++
     #pragma acc exit data delete(ptr[0:num])
     ```

## 実装例

| ソースコード | 実装概要 | 備考 |
| ---- | ---- | ---- |
| [cpp/openacc/0_unified/nbody_leapfrog2.cpp](/cpp/openacc/0_unified/nbody_leapfrog2.cpp) | Unified Memoryを用いた実装，Leapfrog法 | |
| [cpp/openacc/1_data/nbody_leapfrog2.cpp](/cpp/openacc/1_data/nbody_leapfrog2.cpp) | データ指示文を用いた実装，Leapfrog法 | |
| [cpp/openacc/0_unified/nbody_hermite4.cpp](/cpp/openacc/0_unified/nbody_hermite4.cpp) | Unified Memoryを用いた実装，Hermite法 | 一部関数のGPU化を無効化 |
| [cpp/openacc/1_data/nbody_hermite4.cpp](/cpp/openacc/1_data/nbody_hermite4.cpp) | データ指示文を用いた実装，Hermite法 | 一部関数のGPU化を無効化 |

* 一部関数のGPU化について
  * GPU上で動作させるとコードが正常に動作しなくなる関数があったため，暫定的にCPU上で動作させることにしている
    * CUDA版ではGPU上で正常に動作するため，実装ミスやコンパイラのバグなどが原因と考えられる
  * マクロ `EXEC_SMALL_FUNC_ON_HOST` を有効化（= 一部の OpenACC 指示文をコメントアウト）している
  * 小さい関数なので，実行時間への影響も小さいと考えている
  * 余分なCPU-GPU間のデータ転送が生じてしまっている

## NVIDIA HPC SDKに関する情報

### コンパイル・リンク

* 標準的な引数（コンパイル時）

  ```sh
  -acc=gpu -gpu=cc90 -Minfo=accel,opt # OpenACCを使用してGPU化，cc90（NVIDIA H100）向けに最適化，GPUオフローディングや性能最適化に関するコンパイラメッセージを出力
  -acc=gpu -gpu=cc90,mem:unified:nomanagedalloc -Minfo=accel,opt # NVIDIA GH200上でUnified Memoryを使用する際のお勧め設定
  -acc=gpu -gpu=cc90,managed -Minfo=accel,opt # NVIDIA GH200以外の環境（x86 CPUとNVIDIA GPUの組み合わせ）でUnified Memoryを使用する際の設定
  ```

* 標準的な引数（リンク時）

  ```sh
  -acc=gpu     # 指定し忘れると，GPU上では動作しない
  -gpu=mem:unified:nomanagedalloc # Unified Memory 使用時にはこれもつける（NVIDIA GH200）
  -gpu=managed # Unified Memory 使用時にはこれもつける（NVIDIA GH200以外の環境：x86 CPUとNVIDIA GPUの組み合わせ）
  ```

### 実行時

* デバッグ時に便利な環境変数

  ```sh
  NVCOMPILER_ACC_NOTIFY=1 ./a.out # GPU 上でカーネルが実行される度に情報を出力する
  NVCOMPILER_ACC_NOTIFY=3 ./a.out # CPU-GPU 間のデータ転送に関する情報も出力する
  NVCOMPILER_ACC_TIME=1 ./a.out   # CPU-GPU 間のデータ転送および GPU 上での実行時間を出力する
  ```
