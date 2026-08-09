# fwaun-tools

fwaun モデル群の学習を支えるツール群です。1つのバイナリに2つのコマンドグループを備えます:

- **`dataset`** — ローカルの Stable Diffusion LoRA データセット向けのタグ・
  キャプション編集。手動編集／WD14系の自動タガー／Qwen3-VL のキャプショナー／
  Danbooru API からのタグ取得を1つの編集画面に統合し、出所を意識せず
  ひとつのチップ列としてタグを整備できます。
- **`model`** — safetensors チェックポイント操作。タスクベクトルの
  `merge-diff`、LoRA 抽出（`extract-lora`）、INT8+ConvRot 量子化
  （`quant-int8`）。純 Rust/CPU 実装で、どのビルドでも利用できます。

主に [ANIMA preview][anima] と Krea 2 の LoRA 学習向けに作っていますが、
データモデルやエクスポートプロファイルは ANIMA 専用ではありません。

`dataset` コマンド群は [kohya-ss/musubi-tuner][musubi] と
[kohya-ss/sd-scripts][sd-scripts] が扱う入力形式を、`model` コマンド群は
[ComfyUI][comfyui] との連携を前提に設計しています。それ以外の学習・推論環境
での利用はテストしていません。

> [English README](README.md)

[anima]: https://civitai.com/models/anima-preview
[musubi]: https://github.com/kohya-ss/musubi-tuner
[sd-scripts]: https://github.com/kohya-ss/sd-scripts
[comfyui]: https://github.com/Comfy-Org/ComfyUI

## 主な機能

- **3つのソースをひとつのチップ列で編集。** 手動タグ・自動タガー出力・
  Danbooru タグが同じリストに並びます。色で区別はされますが、編集時に
  出所を意識する必要はありません。
- **モデルを切り替えても消えない非表示指定。** `-foo` を一度書いておけば、
  別のタガーモデルで再実行しても抑制設定は維持されます。
- **データセット共通タグを1行で。** `fwaun-tools.toml` の `common_tags` に
  書いたタグは、サイドカーを一切書き換えずにデータセット内の全画像へ適用
  されます（あとから追加した画像にも自動で効きます）。個別画像側での
  打ち消しも可能。
- **整理用タグ（出力されないラベル）。** アンダースコア始まりの手動タグ
  (`_foo`) はデータに残りタググループ分類にも使われますが、書き出しには
  含まれません。「確認済みだがどれでもない」を「未確認」と区別できます。
- **タググループ + カンバン表示。** `fwaun-tools.toml` で排他的なタグの組
  （例: 衣装の種類）を宣言すると、GUI ではタグごとの列にサムネイルが
  振り分けられ、ドラッグ&ドロップで切り替えられます。
- **拡大表示。** サムネイルをダブルクリックすると画面いっぱいに原寸表示され、
  ← / → で絞り込み後の前後の画像へ移動、Esc で閉じます。右クリックメニューの
  **既定のアプリで開く** / **フォルダで表示** から普段のビューアに渡すことも
  できます。
- **データセット単位の設定 (`fwaun-tools.toml`)。** タガーモデル、
  キャプショナー、エクスポートプロファイル、しきい値などをディレクトリごとに切り替え可能。
- **2つの書き出しモード。** `export` は `<image>.txt` を画像ごとに出力
  （sd-scripts の DreamBooth / LoRA caption-file モード）。
  `metadata` はディレクトリ全体で1つの `meta.json` を出力（sd-scripts ファインチューンモード）。
- **開き直しが速い。** GUI の**再読み込み**ボタンは開いているフォルダを再スキャン
  し、実際に変更のあった画像だけサムネイルを作り直します。CLI での一括処理や
  数枚の追加を反映するのに全件の再生成は不要です。セッションをまたぐ場合は
  サムネイルのディスクキャッシュが同じ役割を果たします（上限つき。設定モーダルの
  **アプリ**タブから削除できます）。
- **GUIは日英両対応。** 英語／日本語の切り替え、デフォルトはホストOSのロケール準拠。
- **CLI でバッチ処理、GUI でキュレーション。** GUI には
  **モデルツール**タブ（データセット／モデルツールのモード切り替え）もあり、
  `merge-diff` / `extract-lora` / `quant-int8` のチェックポイント処理を
  CLI を使わずに実行できます。

## インストール

### macOS (Apple Silicon) / Linux (x64 / arm64)

```sh
curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh
```

### Windows (x64)

```powershell
irm https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.ps1 | iex
```

どちらのスクリプトも、最新のGitHubリリースをダウンロードし、SHA256を検証したうえで、CLI と GUI のバイナリを並べて配置します:

| プラットフォーム | CLI                                  | GUI                                          |
| ---------------- | ------------------------------------ | -------------------------------------------- |
| macOS            | `~/.local/bin/fwaun-tools`          | `~/.local/bin/fwaun-tools-gui`              |
| Linux            | `~/.local/bin/fwaun-tools`          | `~/.local/bin/fwaun-tools-gui`              |
| Windows          | `%USERPROFILE%\bin\fwaun-tools.exe` | `%USERPROFILE%\bin\fwaun-tools-gui.exe`     |

特定のバージョンを指定したい場合は `--version v0.2.1` （PowerShellなら `-Version v0.2.1`）を付けてください。

デフォルトでは両方のバイナリをインストールしますが、**ヘッドレスの Linux**
（`$DISPLAY` / `$WAYLAND_DISPLAY` が無い環境）では CLI のみをインストールします
（GUI は実行に画面が必要なため）。`--both` / `--cli-only` / `--gui-only` で
選択を上書きできます:

```sh
# CLI のみ（SSH でしか触らない学習用マシンなど）
curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh -s -- --cli-only
# ヘッドレスでも両方を強制（リモート / 転送ディスプレイを使う場合）
curl -fsSL https://raw.githubusercontent.com/fwaunstp/fwaun-tools/main/install.sh | sh -s -- --both
```

GUI は [egui][egui] でビルドされたシングルバイナリです。
`.app`/AppImage/MSI といった追加ラッパーはありません。
Linux では標準的な X11 / Wayland ライブラリ
（どのデスクトップ環境にも入っている類のもの）に動的リンクされますが、
追加のランタイムインストールは不要です。

macOS のバイナリは **公証 (notarize) されていません**。
インストーラ側で `com.apple.quarantine` 属性は除去しますが、
Finder からの起動を Gatekeeper がブロックする場合は、
ターミナルから一度だけ `~/.local/bin/fwaun-tools-gui` を実行してください。

[egui]: https://github.com/emilk/egui

### Linux の glibc 要件（full ビルドのみ）

Linux 向け**リリース**バイナリは *full* ビルド（[ビルドの種類](#ビルドの種類)参照）
のため、**Ubuntu 24.04 同梱の glibc 2.39** を前提にリンクされています。
Ubuntu 22.04 や Debian 12 以前では起動しません。依存しているローカルタガー /
キャプショナー用の ONNX Runtime プリビルドバイナリが glibc 2.38 で追加された
`__isoc23_*` シンボルを参照しているためです。ディストリビューションを
アップグレードするか、*light* CLI をソースからインストールしてください
（ONNX Runtime を含まず glibc の下限もないため、これらの古い環境でも動きます）:

```sh
cargo install --git https://github.com/fwaunstp/fwaun-tools fwaun-tools-cli
```

`cargo install` はデフォルトの *light* バリアントをビルドします（Rust
ツールチェインが必要。light のプリビルド版は配布していません — 
[ビルドの種類](#ビルドの種類)参照）。`install.sh` は古い glibc を検出すると
同じヒントを表示します。

### Windows サポートについての注意

メンテナは macOS と Linux 中心で開発しています。
Windows バイナリは CI でビルドされていますが、メンテナの手元での動作確認は十分にできていません。
不具合があれば Issue を立てていただけると助かります。

### ソースからビルド

Rust 1.87+ (edition 2024) が必要です。
Linux では GUI のビルドに標準的な X11 / Wayland 開発ヘッダ
(`libx11-dev` / `libxcb1-dev` / `libxkbcommon-dev` / `libwayland-dev` /
`libgl1-mesa-dev` など、ディストリビューションによって名前が異なる場合あり)
が必要です。

```sh
git clone https://github.com/fwaunstp/fwaun-tools
cd fwaun-tools
# light ビルド（デフォルト）— ローカル ONNX 推論なし、どこでも動く
cargo build --release -p fwaun-tools-cli
cargo build --release -p fwaun-tools-gui
# full ビルド — ローカル WD14 タガー + Qwen3-VL キャプショナーを追加（glibc 2.38+）
cargo build --release -p fwaun-tools-cli --features full
cargo build --release -p fwaun-tools-gui --features full
```

### ビルドの種類

`full` cargo feature で 2 種類のビルドを切り替えます:

| | light（デフォルト） | full（`--features full`） |
| --- | --- | --- |
| ローカル WD14 タガー（`tag`） | ✗ | ✓ |
| ローカル Qwen3-VL キャプショナー | ✗ | ✓ |
| OpenAI 互換キャプショナー（`caption`） | ✓ | ✓ |
| booru / export / metadata / 手動編集 / タググループ | ✓ | ✓ |
| ONNX Runtime のリンク | なし | あり |
| Linux の glibc 下限 | なし（古い環境でも動く） | 2.38+ |
| CLI のおおよそのサイズ | 約 11 MB | 約 35 MB |

公開している**リリース**バイナリは *full* です。*light* バイナリはローカル
ONNX モデル 2 種（WD14 タグ付け・Qwen3-VL キャプション）を落としますが、
それ以外はすべて残ります — OpenAI 互換エンドポイント（llama.cpp / Ollama /
LM Studio / vLLM など）経由のキャプションも含みます。light ビルドで
ローカル ONNX 専用コマンドを実行すると、full ビルドを入れるよう促す
メッセージを出して即座に失敗します。API 経由でキャプションする場合や、
glibc の古いホストで動かしたい場合は light を使ってください。

## クイックスタート

1. `fwaun-tools-gui` を起動します（CLI を直接使うこともできます。下のCLIコマンド参照）。
2. **フォルダを開く…** から画像のあるディレクトリを選びます。
3. （任意）**設定…** で `fwaun-tools.toml` を編集します。
   何も設定しなくても妥当なデフォルトで動作します。
4. 画像を選択し、**タガーを実行** / **キャプショナーを実行** / **Booru取得**
   を押します。初回実行時に必要な ONNX モデルが
   HuggingFace のキャッシュ (`~/.cache/huggingface/hub`) にダウンロードされます。
   キャッシュ場所は `HF_HOME` で変更できます。`huggingface.co` に直接アクセスできない
   場合は `HF_ENDPOINT`（例: `https://hf-mirror.com`）でミラーを指定できます。
5. キュレーション作業: 手動タグの追加、不要な自動/Booru タグを `×` で非表示化（取り消し線で表示）、
   キャプションの編集など。
6. ディスクに書き出します:

   ```sh
   fwaun-tools dataset export <dir>          # 画像ごとに .txt を出力
   fwaun-tools dataset metadata <dir>        # 1つの meta.json にまとめて出力
   ```

## 設定ファイルの概要

`fwaun-tools.toml` はデータセットのディレクトリに置きます。
書かなくても動きます（デフォルトが使われます）。
注釈付きのフルスキーマは
[`crates/core/fwaun-tools.toml.example`](crates/core/fwaun-tools.toml.example) を参照してください。
主な項目は以下のとおりです。

```toml
default_profile   = "anima"
default_tagger    = "wd-eva02-large-v3"
default_captioner = "qwen3-vl-4b"

# データセット内の全画像に適用されます（サイドカーには書き込まれません）
common_tags = ["himeko", "-red_hair", "-yellow_eyes"]

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

データセットのキュレーション（`fwaun-tools dataset <verb>`）:

```
fwaun-tools dataset tag <dir>      [--model NAME] [--threshold X] [--force]
fwaun-tools dataset caption <dir>  [--model NAME] [--force]
fwaun-tools dataset booru <dir>    [--source danbooru] [--force]
fwaun-tools dataset export <dir>   [--profile NAME] [--threshold X]
fwaun-tools dataset metadata <dir> [--profile NAME] [--threshold X] [--output PATH]
fwaun-tools dataset add-tag <dir>    --tags TAG[,...] [--dry-run] [--per-image]
fwaun-tools dataset remove-tag <dir> --tags TAG[,...] [--dry-run] [--per-image]
fwaun-tools dataset replace-tag <dir> --from 旧[,...] --to 新[,...] [--dry-run] [--per-image]
fwaun-tools dataset mv <dir> <dest>  --tags TAG[,...] [--dry-run]
fwaun-tools dataset status <dir>
fwaun-tools dataset tokens <dir>
fwaun-tools dataset validate-tag-group <dir> --group NAME [--problems-only] [--json]
```

既存の ComfyUI サーバーを使う一括処理。いずれも結果を別ディレクトリ
（既定では `<dir>_upscaled` / `<dir>_edited` の兄弟ディレクトリ）に書き出し、
`.ron` サイドカーを一緒に複製し、出力が既にある画像はスキップします。
中断した実行はそのまま再開できます:

```
fwaun-tools dataset upscale <dir> [--profile NAME] [--upscale-model FILE] [--workflow JSON] [--max-edge N] [--force] [--dry-run]
fwaun-tools dataset upscale-models     [--profile NAME] [--base-url URL]
fwaun-tools dataset edit <dir>    [--profile NAME] [--prompt TEXT] [--model NAME] [--resolution 1K|2K|4K] [--workflow JSON] [--limit N] [--force] [--dry-run]
fwaun-tools dataset edit-models        [--profile NAME] [--base-url URL]
```

`edit` は Gemini image（Nano Banana）API ノード経由で、1 つの指示を全画像に
適用します。イラストのデータセットの背景だけを実写に差し替える、といった用途
向けです。**このノードは 1 枚ごとに課金される**ため、まず `--dry-run` で枚数を
確認し、`--limit` で数枚だけ試してからバッチを回してください。

また comfy.org のアカウント API キーが必要です。ComfyUI は課金対象の API ノードを
リクエスト単位で認証するため、ブラウザで Web UI にログイン済みでも HTTP API 経由の
実行はカバーされず、キーがないと全画像が *"Please login first to use this node"* で
失敗します。<https://platform.comfy.org> でキーを発行して環境変数に入れてください
（プロファイルの `api_key` でも読めますが、データセット直下の `fwaun-tools.toml` は
データセットと一緒に持ち出されがちなので環境変数を推奨します）:

```sh
export COMFY_API_KEY=comfyui-...        # PowerShell: $env:COMFY_API_KEY='comfyui-...'
```

`tag` / `caption` / `booru` / `upscale` / `edit` は 1 枚ずつ処理するため、実行中は
進捗カウンタを表示します:

```
caption [  12/340 ]   3.5%  1:04 elapsed  ETA 28:37  …_0013.png
```

出力先は stderr で、同じ行を上書きします。1 枚ごとの結果行（stdout）は
従来どおりなので、`… > run.log` にはこれまでと同じ内容だけが残ります。
stderr が端末でない場合（ファイルへのリダイレクト、CI）は上書きをやめ、
全体の 5% ごとに 1 行ずつ出力します。

チェックポイント操作（`fwaun-tools model <verb>`）— データセットではなく
safetensors ファイルを対象にします:

```
fwaun-tools model merge-diff   --base B --tuned T --target G -o OUT [--multiplier M] [--model krea2|anima|auto] [--save-dtype bf16|fp16|fp32]
fwaun-tools model extract-lora --base B --tuned T -o OUT [--rank R] [--alpha A] [--model krea2|anima|auto] [--include RE] [--exclude RE]
fwaun-tools model quant-int8   SRC [DST] [--dry-run] [--include RE] [--exclude RE] [--min-gemm N] [--verify-report PATH]
```

`merge-diff` はフルファインチューンの差分（`tuned − base`）を別の
チェックポイントへ転写し、`extract-lora` はその差分を SVD で kohya-ss/ComfyUI
形式の LoRA に分解、`quant-int8` は comfy-kitchen の `int8_tensorwise` +
ConvRot レイアウトを書き出します。いずれも CPU/f32 でキー単位にストリーム
処理するため、ピーク時のメモリ使用量は小さく保たれます。

これら3つは GUI の**モデルツール**タブ（ウィンドウ上部でモードを切り替え）
からも同じ操作を実行できます。

`quant-int8` の INT8+ConvRot 方式は Comfy-Org の
[`quant_int8_convrot.py`][quant-ref] を参考にしています。

[quant-ref]: https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py

`add-tag` / `remove-tag` はディレクトリ内のマニュアルタグをまとめて編集
します。`add-tag` は各タグをそのまま追加し（`foo` はポジティブ、`-foo`
はサプレッションマーカー）、`remove-tag` は一致するマニュアルタグを
大文字小文字を無視して削除します（サプレッションマーカーを消すには
`--tags=-foo` のように渡します）。

`<dir>` がデータセットルートの場合、この2つは `common_tags` を編集します
（後述の[データセット共通タグ](#データセット共通タグ)を参照）。

`replace-tag` はタグをリネームします。一致したエントリは**その位置のまま**
書き換わるので `manual_tags` の順序が保たれ、旧タグを持たない画像は変更され
ません。`remove-tag` + `add-tag` ではこうはいきません（タグが末尾に移動し、
新タグがディレクトリ全体に付いてしまいます）。

```sh
fwaun-tools dataset replace-tag ./dataset --from long_hair --to very_long_hair
# 対応表を一括で流す — N 番目の --from と N 番目の --to が対になります
fwaun-tools dataset replace-tag ./dataset --from cape,hat --to red_cape,straw_hat --dry-run
```

一致判定は `remove-tag` と同じで、大文字小文字を無視し、先頭の `-` は
エントリの一部として扱います（`--from=-foo --to=-bar` でサプレッション
マーカーをリネーム、`--from Foo --to foo` で表記ゆれの修正）。新タグを既に
持っている画像では、リネームされたエントリは重複せずそこに畳まれます。
他の2つと違い常にサイドカーを走査し、データセットルートでは `common_tags`
のエントリも合わせてリネームします（`--per-image` で設定ファイルは対象外）。

GUI にも選択範囲に対する同じ操作があります。一括パネルの手動タグチップを
右クリックして **このタグをリネーム…**、またはチップ下のリネーム欄に
旧タグ・新タグを入力します。

## データセット共通タグ

キャラクター LoRA では、全画像に同じ手動エントリを入れたくなります。
キャラ名のトリガーワードと、そのトリガーに吸収させたい特徴
（髪色・目色など）の `-foo` 抑制です。これを一箇所に書きます:

```toml
common_tags = ["himeko", "-red_hair", "-yellow_eyes", "-slit_pupils"]
```

記法はサイドカーの `manual_tags` と同じです（`foo` 通常、`-foo` 抑制、
`_foo` 整理用）。このリストは各画像の手動エントリの下に敷かれる仮想
レイヤーとして扱われ、**サイドカーには一切書き込まれません**。そのため
あとから画像を追加しても再実行は不要で、この行を消せば全画像から一括で
外れます。共通レイヤーの通常タグは先頭に出力されるので、トリガーワードが
書き出しの先頭に来ます。

個別の画像は、同じタグを自分で指定することで打ち消せます。画像側の
`red_hair` はデータセット全体の `-red_hair` をその画像だけで無効化し、
画像側の `-himeko` はそのキャラが写っていないカットからトリガーを外します。

このレイヤーは画像のタグを読むすべての箇所で効きます — `export`、
`metadata`、キャプションの prefix/suffix、タググループ分類、カンバン表示、
`mv --tags`。

### リストの編集

データセット全体にかかる一括タグ編集は、データセットについての指定なので、
サイドカーではなく設定ファイルに書き込まれます:

```sh
# <dir> に fwaun-tools.toml がある → .ron ではなく common_tags を書き換え
fwaun-tools dataset add-tag ./dataset --tags=himeko,-red_hair
fwaun-tools dataset remove-tag ./dataset --tags=-red_hair
```

**サブディレクトリ**はデータセットの一部でしかないので、従来どおり
サイドカーを書き換えます。ルートでも従来の挙動にしたい場合は
`--per-image` を付けます。`--dry-run` は設定ファイルを書かずに結果だけ
表示します。ルートでの `remove-tag` は個別画像側のコピーには手を付けず
（意図的な打ち消しの可能性があるため）、該当件数を知らせます。

GUI でも同じルールです。設定ファイルを持つフォルダで**全画像**を選択して
いるときのタグ追加・削除は `fwaun-tools.toml` を編集します。1枚でも選択を
外せばサイドカー編集に戻ります。共通レイヤー由来のタグは専用の色のチップ
（単一画像表示では `[C]` 付き）で表示され、クリックすると選択中の画像だけ
で打ち消せます。

この経路の設定ファイル書き換えはピンポイントで行われるため、手書きの
`fwaun-tools.toml` のコメントやレイアウトはそのまま残ります。

## タググループ

排他的なタグの組を名前付きで宣言する仕組みです。CLI の
`validate-tag-group` は各画像をグループ内のいずれかのタグ・「未設定」・
「違反」（複数のグループタグが共存している状態 — エラーではなく情報表示）
のいずれかに分類して一覧します。GUI の **表示 → カンバン** モードでは
同じバケットを列として描画し、サムネイルをドラッグ&ドロップすると
`manual_tags` を書き換えてタグを切り替えます。

キャラクター LoRA の衣装ごとの分類例。各画像が必ずどれかの列に入るので、
分類漏れに気付きやすくなります:

```toml
[tag_group.official_costumes]
tags = ["official_school_uniform", "official_lounge_wear"]
```

タグ1つだけのグループも有効です。「特定のタグが設定済みか」を
確認したいだけのときに便利です:

```toml
[tag_group.solo_check]
tags = ["solo"]
```

### 整理用（出力されない）タグ

アンダースコア始まりの手動タグ (`_foo`) は **整理用タグ** として扱われ、
データには残りタググループ分類にも使われますが、書き出される `.txt` には
一切含まれません。（非表示指定は `-foo`、整理用は `_foo` です。）

これはキャラクター・画風のタグ付でよくある曖昧さを解決します。グループ
タグが付いていない画像は「まだ確認していない」のか「確認済みだがどれでも
ない」のか区別できませんが、整理用タグをグループのメンバーに加えると、
後者を「未設定」とは別のカンバン列に分けられます:

```toml
[tag_group.character]
tags = ["character_a", "character_b", "_no_character"]
```

`_no_character` 列にドラッグした画像は、アンダースコアタグが学習用
キャプションに漏れることなく「確認済み」として記録されます。

```sh
fwaun-tools dataset validate-tag-group ./dataset --group official_costumes
```

## ドキュメント

- **[DEVELOPMENT.md](DEVELOPMENT.md)** （英語のみ） — 内部アーキテクチャ、
  クレート構成、ONNX セッションの形状、ort バージョン関連の注意点など。
  コードに手を入れる前に一読することを推奨します。
- **[crates/core/fwaun-tools.toml.example](crates/core/fwaun-tools.toml.example)** —
  注釈付きの設定ファイル例。

## プロジェクトの状況とコントリビューション

これは、作者が自分のモデル — Civitai の [fwaun-anima][m-anima]、
[fwaun-krea2][m-krea2]、[fwaun-style][m-style] — を作るために個人的に開発
しているツールです。テストは十分ではなく、不具合が多数残っている可能性が
あります。

新機能の追加は確約できません（作者自身の作業に必要と判断すれば追加します
が、そうでなければ対応しないことがあります）。一方で、既存機能のバグ修正
には積極的に対応するつもりですので、イシューやプルリクエストは歓迎します。

[m-anima]: https://civitai.red/models/2602206/fwaun-anima
[m-krea2]: https://civitai.red/models/2757203/fwaun-krea2
[m-style]: https://civitai.red/models/2593831/fwaun-style

## ライセンス

以下のいずれか、利用者の選択により使用できます:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)
