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
