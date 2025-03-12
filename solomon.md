# Solomon 実装の概要

[Solomon](https://github.com/ymiki-repo/solomon) を用いた$N$体計算コード（直接法）の実装概要の紹介

## Solomon の概要

* Solomon (Simple Off-LOading Macros Orchestrating multiple Notations) は，GPU 向けの指示文である OpenACC や OpenMP target のインタフェースを統合したマクロライブラリです
  * [OpenACC](openacc.md) と [OpenMP target](openmp.md) の両方に対応したコードを簡易に実装できるようになります
  * 実装の概要は [Solomonのリポジトリ](https://github.com/ymiki-repo/solomon) 中の [日本語版README](https://github.com/ymiki-repo/solomon/blob/main/README_jp.md) を参照してください
  * 実装の詳細は [Miki & Hanawa (2024, IEEE Access, vol. 12, pp. 181644-181665)](https://doi.org/10.1109/ACCESS.2024.3509380) を参照してください
    * Solomon を利用した論文においては，[Miki & Hanawa (2024, IEEE Access, vol. 12, pp. 181644-181665)](https://doi.org/10.1109/ACCESS.2024.3509380) を引用してください

## 実装方法

* Solomon のヘッダファイルをインクルード

   ```c++
   #include <solomon.hpp>
   ```

* GPU上で計算させたいfor文に指示文を追加

   ```c++
   OFFLOAD(AS_INDEPENDENT)
   for(int32_t ii = 0; ii < num; ii++){
     ...
   }
   ```

  * オプション：スレッドブロックあたりのスレッド数を示唆

     ```c++
     OFFLOAD(AS_INDEPENDENT, NUM_THREADS(256))
     for(int32_t ii = 0; ii < num; ii++){
       ...
     }
     ```

* （Unified Memoryを使わない場合）データ指示文を追加
  1. GPU上のメモリ確保

     ```c++
     MALLOC_ON_DEVICE(ptr[0:num])
     ```

  2. CPUからGPUへのデータ転送

     ```c++
     MEMCPY_H2D(ptr[0:num])
     ```

  3. GPUからCPUへのデータ転送

     ```c++
     MEMCPY_D2H(ptr[0:num])
     ```

  4. GPU上のメモリ解放

     ```c++
     FREE_FROM_DEVICE(ptr[0:num])
     ```

## 実装例

| ソースコード | 実装概要 | 備考 |
| ---- | ---- | ---- |
| [cpp/solomon/nbody_leapfrog2.cpp](/cpp/solomon/nbody_leapfrog2.cpp) | データ指示文を用いた実装，Leapfrog法 | |
| [cpp/solomon/nbody_hermite4.cpp](/cpp/solomon/nbody_hermite4.cpp) | データ指示文を用いた実装，Hermite法 | 一部関数のGPU化を無効化 |

* 一部関数のGPU化について
  * GPU上で動作させるとコードが正常に動作しなくなる関数があったため，暫定的にCPU上で動作させることにしている
    * CUDA版ではGPU上で正常に動作するため，実装ミスやコンパイラのバグなどが原因と考えられる
  * マクロ `EXEC_SMALL_FUNC_ON_HOST` を有効化（= 一部の OpenACC 指示文をコメントアウト）している
  * 小さい関数なので，実行時間への影響も小さいと考えている
  * 余分なCPU-GPU間のデータ転送が生じてしまっている

## コンパイル方法

* コンパイラオプションを用いて，[OpenACC](openacc.md) または [OpenMP target](openmp.md) を有効化してください
* Solomon のパス（`solomon.hpp` があるディレクトリ）を `-I/path/to/solomon` などとして指示してください
* 下記のコンパイルフラグを用いて，Solomon の動作モードを指定してください

  | コンパイルフラグ | 使用されるバックエンド | 備考 |
  | ---- | ---- | ---- |
  | `-DOFFLOAD_BY_OPENACC` | OpenACC | デフォルトでは `kernels` 構文を使用 |
  | `-DOFFLOAD_BY_OPENACC -DOFFLOAD_BY_OPENACC_PARALLEL` | OpenACC | デフォルトでは `parallel` 構文を使用 |
  | `-DOFFLOAD_BY_OPENMP_TARGET` | OpenMP target | デフォルトでは `loop` 指示文を使用 |
  | `-DOFFLOAD_BY_OPENMP_TARGET -DOFFLOAD_BY_OPENMP_TARGET_DISTRIBUTE` | OpenMP target | デフォルトでは `distribute` 指示文を使用 |
  | | 縮退モード | OpenMP を用いたマルチコアCPU向けのスレッド並列 |

  * 注意： コンパイルフラグとして `-DOFFLOAD_BY_OPENACC` を渡しても，使用するコンパイラに対して OpenACC を有効にするフラグ（例： NVIDIA HPC SDKにおいては`-acc`）を渡していない場合には，`-DOFFLOAD_BY_OPENACC` については自動的に無効化されます
