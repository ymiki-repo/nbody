# OpenACC 実装の概要

OpenACC を用いた$N$体計算コード（直接法）の実装概要の紹介

## NVIDIA HPC SDK使用時に便利な環境変数

```sh
NVCOMPILER_ACC_NOTIFY=1 ./a.out # GPU 上でカーネルが実行される度に情報を出力する
NVCOMPILER_ACC_NOTIFY=3 ./a.out # CPU-GPU 間のデータ転送に関する情報も出力する
NVCOMPILER_ACC_TIME=1 ./a.out # CPU-GPU 間のデータ転送および GPU 上での実行時間を出力する
```
