# OpenACC 実装の概要

OpenACC を用いた$N$体計算コード（直接法）の実装概要の紹介

## 実装方法

* GPU上で計算させたいfor文に指示文を追加

   ```c++
   #pragma acc kernels
   #pragma acc loop independent
   for(int32_t ii = 0; ii < num; ii++){
     ...
   }
   ```

  * オプション：スレッドブロックあたりのスレッド数を示唆

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

## NVIDIA HPC SDKに関する情報

### コンパイル・リンク

* 標準的な引数（コンパイル時）

  ```sh
  -acc=gpu -gpu=cc80 -Minfo=accel,opt # OpenACCを使用してGPU化，cc80（NVIDIA A100）向けに最適化，GPUオフローディングや性能最適化に関するコンパイラメッセージを出力
  -acc=gpu -gpu=cc80,managed -Minfo=accel,opt # 上記に加えて，Unified Memoryを使用
  ```

* 標準的な引数（リンク時）

  ```sh
  -acc=gpu
  ```

### 実行時

* デバッグ時に便利な環境変数

  ```sh
  NVCOMPILER_ACC_NOTIFY=1 ./a.out # GPU 上でカーネルが実行される度に情報を出力する
  NVCOMPILER_ACC_NOTIFY=3 ./a.out # CPU-GPU 間のデータ転送に関する情報も出力する
  NVCOMPILER_ACC_TIME=1 ./a.out   # CPU-GPU 間のデータ転送および GPU 上での実行時間を出力する
  ```