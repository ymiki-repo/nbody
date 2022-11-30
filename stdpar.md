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

  * counting_iterator については別途自分で実装する，thrustなどから呼ぶという方針もあるが，ここでは作業量をなるべく減らす，可搬性（NVIDIA HPC SDK以外のコンパイラでもコンパイルできるようにしておく）ためにBoost C++を利用することとした

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
