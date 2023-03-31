# Goで「ゼロから作るDeep Learning」

[オライリーの「ゼロから作るDeep Learning」（斎藤康毅著）](https://www.oreilly.co.jp/books/9784873117584/)をGo言語でやっていくリポジトリ。

この本ではGPU上で実行するための実装については扱われていないが、本リポジトリではOpenCLを使った実装にもチャレンジする。

なお、OpenCLで標準的に扱えるのは`float32`（`float`、単精度浮動小数点数）型で、[Gonum](https://pkg.go.dev/gonum.org/v1/gonum)で扱えるのは`float64`（`double`倍精度浮動小数点数）型と異なっているため、CPU実装とGPU実装の間に互換性はない。

## CPU

[Gonum](https://pkg.go.dev/gonum.org/v1/gonum)を利用して、倍精度浮動小数点数で計算する。

## GPU

[go-opencl](https://github.com/PassKeyRa/go-opencl)を利用して、単精度浮動小数点数で計算する。

## ビルド

```sh
# CPU
go build

# GPU (needs OpenCL runtime)
go build --tags gpu
```

例えば、WindowsでAMD APP SDK 3.0のランタイムを使用する場合は以下のようにします。

```ps1
Set-Item -Path env:CGO_CFLAGS -Value "-I C:\PROGRA~2\AMDAPP~1\3.0\include"
Set-Item -Path env:CGO_LDFLAGS -Value "-L C:\PROGRA~2\AMDAPP~1\3.0\lib\x86_64"
```

## テスト

```sh
# CPU
go test --tags cpu ./...

# GPU
go test --tags gpu ./...
```
