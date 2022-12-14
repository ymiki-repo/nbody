# C++17の標準言語規格を用いた実装の概要

C++17の標準言語規格を用いた$N$体計算コード（直接法）の実装概要の紹介

## 実装方法

* algorithm, execution を include

   ```c++
   #include <algorithm>
   #include <execution>
   ```

* GPU上で計算させたいfor文をstd::for_each_nに置換

   ```c++
   std::for_each_n(std::execution::par, boost::iterators::counting_iterator<int32_t>(0), num, [=](const int32_t ii){
     ...
   });
   ```

  * 実装方法は他にもあるが，ある程度長いfor文については上記の方法が最短コース
    * 並列版アルゴリズムが提供されている関数については，std::execution::parを追加するだけで良い

       ```c++
       std::sort(std::execution::par, begin(), end());
       ```

  * counting_iterator の実装
    * 取り得る実装方針
      * 別途自分で実装する
      * thrustなどのライブラリから呼ぶ
    * ここではBoost C++から呼ぶこととした
      * 作業量をなるべく減らすため，ライブラリ使用を選択
      * 可搬性（NVIDIA HPC SDK以外のコンパイラでもコンパイルできるようにしておく）ためにthrustは対象外とした

## 実装例

| ソースコード | 実装概要 | 備考 |
| ---- | ---- | ---- |
| [cpp/stdpar/nbody_leapfrog2.cpp](/cpp/stdpar/nbody_leapfrog2.cpp) | Leapfrog法 | |
| [cpp/stdpar/nbody_hermite4.cpp](/cpp/stdpar/nbody_hermite4.cpp) | Hermite法 | 一部関数のGPU化を無効化 |

* 一部関数のGPU化について
  * GPU上で動作させるとコードが正常に動作しなくなる関数があったため，暫定的にCPU上で動作させることにしている
    * CUDA版ではGPU上で正常に動作するため，実装ミスやコンパイラのバグなどが原因と考えられる
  * マクロ EXEC_SMALL_FUNC_ON_HOST を有効化（= 一部で std::execution::par ではなく std::execution::seq を指定して並列化を抑止）している
  * 小さい関数なので，実行時間への影響も小さいと考えている
  * 余分なCPU-GPU間のデータ転送が生じてしまっている

## NVIDIA HPC SDKに関する情報

### コンパイル・リンク

* 標準的な引数（コンパイル時）

  ```sh
  -stdpar=gpu -gpu=cc80 -Minfo=accel,opt,stdpar # 標準言語規格を使用してGPU化，cc80（NVIDIA A100）向けに最適化，GPUオフローディングや性能最適化に関するコンパイラメッセージを出力
  -stdpar=multicore -Minfo=opt,stdpar # 標準言語規格を使用してマルチコアCPU向けに並列化，性能最適化に関するコンパイラメッセージを出力
  ```

* 標準的な引数（リンク時）

  ```sh
  -stdpar=gpu # GPU化した場合
  ```
