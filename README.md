# Goで「ゼロから作るDeep Learning」

[オライリーの「ゼロから作るDeep Learning」（斎藤康毅著）](https://www.oreilly.co.jp/books/9784873117584/)をGo言語でやっていくリポジトリ。

この本では扱われていないが、OpenCLによる実装にもチャレンジする。

## ビルド

### CPU

```sh
go build
```

### GPU

OpenCLのランタイムが必要です。`CGO_CFLAGS`と`CGO_LDFLAGS`を設定してください。

```sh
go build --tags gpu
```

例えば、WindowsでAMD APP SDK 3.0のランタイムを使用する場合は以下のようにします。

```ps1
Set-Item -Path env:CGO_CFLAGS -Value "-I C:\PROGRA~2\AMDAPP~1\3.0\include"
Set-Item -Path env:CGO_LDFLAGS -Value "-L C:\PROGRA~2\AMDAPP~1\3.0\lib\x86_64"
```

## テスト

### CPU

```
go test --tags cpu ./...
```

### GPU

```
go test --tags gpu ./...
```
