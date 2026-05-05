# anima-tagger

ローカルの Stable Diffusion LoRA データセット向け、タグ・キャプション編集ツールです。
手動編集／WD14系の自動タガー／Qwen3-VL のキャプショナー／Danbooru API からのタグ取得を、
1つの編集画面に統合しています。出所を意識せずに、ひとつのチップ列としてタグを整備できます。

主に [ANIMA preview][anima] の LoRA 学習向けに作っていますが、
データモデルやエクスポートプロファイルは ANIMA 専用ではありません。

> [English README](README.md)

[anima]: https://civitai.com/models/anima-preview

## 主な機能

- **3つのソースをひとつのチップ列で編集。** 手動タグ・自動タガー出力・
  Danbooru タグが同じリストに並びます。色で区別はされますが、編集時に
  出所を意識する必要はありません。
- **モデルを切り替えても消えない非表示指定。** `-foo` を一度書いておけば、
  別のタガーモデルで再実行しても抑制設定は維持されます。
- **データセット単位の設定 (`anima-tagger.toml`)。** タガーモデル、
  キャプショナー、エクスポートプロファイル、しきい値などをディレクトリごとに切り替え可能。
- **2つの書き出しモード。** `export` は `<image>.txt` を画像ごとに出力
  （sd-scripts の DreamBooth / LoRA caption-file モード）。
  `metadata` はディレクトリ全体で1つの `meta.json` を出力（sd-scripts ファインチューンモード）。
- **GUIは日英両対応。** 英語／日本語の切り替え、デフォルトはホストOSのロケール準拠。
- **CLI でバッチ処理、GUI でキュレーション。**

## インストール

### macOS (Apple Silicon) / Linux (x64 / arm64)

```sh
curl -fsSL https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.sh | sh
```

### Windows (x64)

```powershell
irm https://raw.githubusercontent.com/fwaunstp/anima-tagger/main/install.ps1 | iex
```

どちらのスクリプトも、最新のGitHubリリースをダウンロードし、SHA256を検証したうえで、CLI と GUI のバイナリを並べて配置します:

| プラットフォーム | CLI                                  | GUI                                          |
| ---------------- | ------------------------------------ | -------------------------------------------- |
| macOS            | `~/.local/bin/anima-tagger`          | `~/.local/bin/anima-tagger-gui`              |
| Linux            | `~/.local/bin/anima-tagger`          | `~/.local/bin/anima-tagger-gui`              |
| Windows          | `%USERPROFILE%\bin\anima-tagger.exe` | `%USERPROFILE%\bin\anima-tagger-gui.exe`     |

特定のバージョンを指定したい場合は `--version v0.1.0` （PowerShellなら `-Version v0.1.0`）を付けてください。

GUI は [egui][egui] でビルドされたシングルバイナリです。
`.app`/AppImage/MSI といった追加ラッパーはありません。
Linux では標準的な X11 / Wayland ライブラリ
（どのデスクトップ環境にも入っている類のもの）に動的リンクされますが、
追加のランタイムインストールは不要です。

macOS のバイナリは **公証 (notarize) されていません**。
インストーラ側で `com.apple.quarantine` 属性は除去しますが、
Finder からの起動を Gatekeeper がブロックする場合は、
ターミナルから一度だけ `~/.local/bin/anima-tagger-gui` を実行してください。

[egui]: https://github.com/emilk/egui

### Linux の glibc 要件

Linux 向けリリースバイナリは **Ubuntu 24.04 同梱の glibc 2.39** を前提に
リンクされています。Ubuntu 22.04 や Debian 12 以前では起動しません。
依存している ONNX Runtime のプリビルドバイナリが glibc 2.38 で
追加された `__isoc23_*` シンボルを参照しているためです。
古いディストリビューションを使う場合はソースからビルドするか、
ディストリビューションをアップグレードしてください。

### Windows サポートについての注意

メンテナは macOS と Linux 中心で開発しています。
Windows バイナリは CI でビルドされていますが、メンテナの手元での動作確認は十分にできていません。
不具合があれば Issue を立てていただけると助かります。

### ソースからビルド

Rust 1.85+ (edition 2024) が必要です。
Linux では GUI のビルドに標準的な X11 / Wayland 開発ヘッダ
(`libx11-dev` / `libxcb1-dev` / `libxkbcommon-dev` / `libwayland-dev` /
`libgl1-mesa-dev` など、ディストリビューションによって名前が異なる場合あり)
が必要です。

```sh
git clone https://github.com/fwaunstp/anima-tagger
cd anima-tagger
cargo build --release -p anima-tagger-cli
cargo build --release -p anima-tagger-gui
```

## クイックスタート

1. `anima-tagger-gui` を起動します（CLI を直接使うこともできます。下のCLIコマンド参照）。
2. **フォルダを開く…** から画像のあるディレクトリを選びます。
3. （任意）**設定…** で `anima-tagger.toml` を編集します。
   何も設定しなくても妥当なデフォルトで動作します。
4. 画像を選択し、**タガーを実行** / **キャプショナーを実行** / **Booru取得**
   を押します。初回実行時に必要な ONNX モデルが
   HuggingFace のキャッシュ (`~/.cache/huggingface/hub`) にダウンロードされます。
5. キュレーション作業: 手動タグの追加、不要な自動/Booru タグを `×` で非表示化（取り消し線で表示）、
   キャプションの編集など。
6. ディスクに書き出します:

   ```sh
   anima-tagger export <dir>          # 画像ごとに .txt を出力
   anima-tagger metadata <dir>        # 1つの meta.json にまとめて出力
   ```

## 設定ファイルの概要

`anima-tagger.toml` はデータセットのディレクトリに置きます。
書かなくても動きます（デフォルトが使われます）。
注釈付きのフルスキーマは
[`crates/core/anima-tagger.toml.example`](crates/core/anima-tagger.toml.example) を参照してください。
主な項目は以下のとおりです。

```toml
default_profile   = "anima"
default_tagger    = "wd-eva02-large-v3"
default_captioner = "qwen3-vl-4b"

[export.anima]
threshold = 0.35
shuffle = false
category_prefixes = { artist = "@" }

[tagger.wd-eva02-large-v3]
repo = "SmilingWolf/wd-eva02-large-tagger-v3"
input_size = 448
storage_threshold = 0.10

[captioner.qwen3-vl-4b]
repo = "onnx-community/Qwen3-4B-VL-ONNX"
subdir = "qwen3-vl-4b-instruct-onnx-vision-fp32-text-int4-cpu"
prompt = "Describe this image in detail."
```

## CLI コマンド

```
anima-tagger tag <dir>      [--model NAME] [--threshold X] [--force]
anima-tagger caption <dir>  [--model NAME] [--force]
anima-tagger booru <dir>    [--source danbooru] [--force]
anima-tagger export <dir>   [--profile NAME] [--threshold X]
anima-tagger metadata <dir> [--profile NAME] [--threshold X] [--output PATH]
anima-tagger status <dir>
anima-tagger tokens <dir>
```

## ドキュメント

- **[DEVELOPMENT.md](DEVELOPMENT.md)** （英語のみ） — 内部アーキテクチャ、
  クレート構成、ONNX セッションの形状、ort バージョン関連の注意点など。
  コードに手を入れる前に一読することを推奨します。
- **[crates/core/anima-tagger.toml.example](crates/core/anima-tagger.toml.example)** —
  注釈付きの設定ファイル例。

## ライセンス

以下のいずれか、利用者の選択により使用できます:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)
