# ゲートウェイへの接続

ゲートウェイ（[local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)）を起動したら、
クライアントは公開ポート（既定 `http://127.0.0.1:8799/v1`）に繋ぎ、リクエストの `model` で使う
モデルを選ぶ。`model` は `gateway.toml` に登録済みのものを指定する（初回リクエストで遅延ロード）。
ローカル（認証なし）なら `api_key` は任意（`"not-needed"` 等で可）。

## 別PC（ネットワーク越し）から繋ぐ

ゲートウェイ側で `host = "0.0.0.0"` と `api_key` を設定して公開している場合（→ サーバー側の
[docs/gateway.md「別PCから接続する」](https://github.com/ToPo-ToPo-ToPo/local-llm-server/blob/main/docs/gateway.md)）、
クライアントは **`base_url` をゲートウェイPCのLAN IP** に、**`api_key` をそのキー** に合わせる。
`api_key` は chat だけでなく在席セッション（即時アンロード）にも自動で載る。

```python
from local_llm_client import LLMClient

llm = LLMClient(
    model="mlx-community/Qwen3.6-27B-4bit",
    base_url="http://192.168.1.5:8799/v1",   # ゲートウェイPCのIP
    api_key="＜gateway.toml の api_key と同じ値＞",
)
```

`api_key` が無い/違うと `401`。`/admin/*`（状態・設定変更）はゲートウェイPC本体からのみで、リモート
からは使えない（chat と在席セッションだけがリモート可）。

## `LLMClient`（推奨）

公式 `openai` SDK を土台にした高レベルクライアント。`respond()` は非ストリームで生成テキスト
（`str`）、`stream=True` で断片の `Iterator[str]` を返す。

```python
from local_llm_client import LLMClient

llm = LLMClient(model="mlx-community/Qwen3.6-27B-4bit",
                base_url="http://127.0.0.1:8799/v1")

print(llm.respond("ローカルLLMの利点を3つ。"))                 # 非ストリーム → str

for piece in llm.respond("もっと詳しく", stream=True):          # ストリーム → Iterator[str]
    print(piece, end="", flush=True)

llm.respond("これは何？", images=["plot.png"])                  # 画像（マルチモーダル）
```

主な引数: `model` / `base_url` / `api_key` / `temperature` / `max_tokens` / `timeout`。
`respond()` は `system_prompt` / `images` / `stream` のほか、追加の `**kwargs` を
`chat.completions.create` にそのまま渡す。

### 起動確認付きのワンライナー `connect()`

`connect()` は接続前にゲートウェイの死活を確認し、繋がった `LLMClient` を返す。未起動なら
`ServerNotRunningError` を投げる（**サーバーは自前で起動しない**）。

```python
from local_llm_client import connect, ServerNotRunningError

try:
    llm = connect(model="mlx-community/Qwen3.6-27B-4bit",
                  base_url="http://127.0.0.1:8799/v1")
except ServerNotRunningError:
    print("先にゲートウェイ（local-llm-server）を起動してください")
```

### 高度な操作（`llm.openai`）

`llm.openai` で土台の openai クライアントに直接アクセスできる。embeddings / tool calling /
構造化出力（`response_format`）/ async など、`respond()` に無い操作はこちらを使う。

```python
emb = llm.openai.embeddings.create(model="...", input="...")
```

## 他の OpenAI 互換クライアント

ゲートウェイは標準的な OpenAI 互換 API なので、`openai` SDK を直接使ったり、他言語の
クライアント、`curl` でもそのまま繋がる（このパッケージを入れなくてもよい）。

### `openai` SDK を直接

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8799/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="mlx-community/Qwen3.6-27B-4bit",
    messages=[{"role": "user", "content": "こんにちは"}],
)
print(resp.choices[0].message.content)
```

### `curl`

```bash
curl -s http://127.0.0.1:8799/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.6-27B-4bit",
    "messages": [{"role": "user", "content": "俳句を一つ詠んでください。"}]
  }' | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

## TLS と実行環境（eMudhra 対応）

2026-10-27 に、このリポジトリの実環境を調査・更新した。実行結果は macOS / Apple Silicon 上のもの。
ライブラリの API、OpenAI SDK の対応範囲 `>=1.55.3,<4`、ロックの OpenAI 2.44.0 は維持した。

変更ファイルは `pyproject.toml`、`uv.lock`（要件・ロック）、
`.vscode/settings.json`、`.gitignore`（環境選択・退避環境とログの管理）、
`scripts/check-fresh-venv.sh`、`scripts/check-tls.py`（検証とログ収集）、
`examples/connect_and_generate.py`、`examples/generate_with_mtp.py`（起動経路）、
`README.md`、本書（手順・結果）の計 10 件。

### 最小要件と開発環境

| 対象 | 必要な最小版 | 開発環境の使用・実測版 |
| --- | --- | --- |
| CPython 3.11 | 3.11.18 | 互換性検証に 3.11.18 を使用 |
| CPython 3.12 | 3.12.16 | 実行未確認、修正範囲は公式情報で確認 |
| CPython 3.13 | 3.13.18 | `uv venv --python 3.13.18` で作成した `.venv` |
| CPython 3.14 | 3.14.10 | 実行未確認、修正範囲は公式情報で確認 |
| CPython 3.15 | 3.15.2 | 実行未確認、修正範囲は公式情報で確認 |
| certifi | 2026.10.19 | `uv.lock`・`.venv` とも 2026.10.19 |
| uv | 鍵拒否は 0.13.5、上記 Python のカタログは 0.13.6 | 通常の Homebrew 版 uv 0.13.6 |
| pip（使用する場合） | 26.2.2 | `.venv` には入れず uv を使用。新 Python の ensurepip は 26.2.2、内蔵 certifi は 2026.10.19 |

`Requires-Python` は各系列の未修正パッチ版を除外する。古いパッチ版の利用者は同系列の
修正版へ更新する必要がある。開発環境の Python の版指定は、公開パッケージの対応範囲とは別に管理する。
Python 3.13.10 から 3.13.18 への更新で系列は変更していない。uv 0.11 系は影響範囲に含まれるため
0.13 系へ変更した。初回対応時のロック変更は certifi のみ。0.10.3 での追加対応は下記を参照。
新 Python の OpenSSL は 3.5.9。`.zshrc`、pyenv の既定、OS の CA は変更していない。
従来の手順を維持するため、通常の `uv python install 3.13.18`、`uv venv --python 3.13.18`、
`uv sync --locked` で構築し、`uv run` で実行する。Python とキャッシュは uv 標準の場所を使う。
手順を戻す時点で Homebrew の uv は既に 0.13.6 だったため、追加の Homebrew 更新は行っていない。
当初作成した `scripts/uv` は削除した。`.python-version` は pyenv とも共有されるため追加しない。
例の独立スクリプト用依存宣言も削除し、開発環境のロックを経由して実行するよう案内した。

### 通信経路

| 経路 | TLS の実体と証明書ストア | 対応 |
| --- | --- | --- |
| 起動確認、モデル一覧、在席セッション | `urllib.request` → CPython ssl/OpenSSL。macOS の `/private/etc/ssl/cert.pem` | Python の侵害鍵拒否を含む版へ更新 |
| OpenAI 1・2 系列の生成・ストリーム・音声送信 | httpx → httpcore → CPython ssl/OpenSSL。既定 CA は certifi | certifi の直接最小要件とロックを更新 |
| OpenAI 3 系列の同操作 | httpx2 → httpcore2 → truststore。既定は OS の証明書検証 | 別環境で実通信を検証。macOS の失効拒否を確認 |
| uv のパッケージ・Python 取得 | Rust の独自 TLS と内蔵 CA | uv 本体を更新。Python/certifi の更新で直るとは扱わない |
| URL 指定の画像・動画の取得 | このクライアントは URL を渡すだけ。取得するゲートウェイは別リポジトリ | ゲートウェイ側の HTTPS は未確認 |

独自 HTTP クライアントへの差し替え、追加 SDK extras、独自 CA、環境変数で選択される別 CA は
今回の検証対象外。OS の失効対策が Python 標準 TLS へ自動的に反映されるとは限らない。
調査開始時のシェルは pyenv Python 3.13.9 / pip 25.3、Homebrew uv 0.11.2 だった。
現在は通常の Homebrew uv 0.13.6 と、このリポジトリの `.venv` を使う。

### 検証方法と結果

`scripts/check-tls.py` は各実装で `https://julialang.org/` の HTTP 200 を先に確認する。
続いて公表元の 12 チェーン（DV/OV/EV × G3B/G3C × emSign/AAA）と No CT の 2 件を検査する。
認証情報を送らず、リダイレクトを追わず、TLS 検証を有効にする。公表元のスクリプトは実行しない。
接続した場合は chains.zip にあるリーフと中間証明書の SHA-256 指紋を照合する。
DNS・タイムアウト・正常対照失敗は未確認とし、期限切れ・ホスト名不一致・CT 要件による拒否を
侵害 CA の拒否とは区別する。終了コードに加えて例外の原因・証明書エラーコードを JSON に記録する。

| 環境 | 既存テスト | urllib の 14 件 | SDK 既定 HTTP 層の 14 件 |
| --- | --- | --- | --- |
| 更新前 Python 3.13.10 / OpenAI 2.44.0 / certifi 2026.6.17 | 97 成功 | 14 件受け入れ | emSign 7 件受け入れ、AAA 7 件拒否 |
| 更新後 Python 3.13.18 / OpenAI 2.44.0 / certifi 2026.10.19 | 97 成功 | 14 件拒否 | 14 件拒否（httpx 0.28.1 / httpcore 1.0.9） |
| Python 3.11.18 / OpenAI 1.55.3 / certifi 2026.10.19 | 97 成功 | 14 件拒否 | 14 件拒否（httpx 0.28.1） |
| Python 3.11.18 / OpenAI 2.54.0 / certifi 2026.10.19 | 97 成功 | 14 件拒否 | 14 件拒否（httpx 0.28.1） |
| Python 3.11.18 / OpenAI 3.0.0 / certifi 2026.10.19 | 97 成功 | 14 件拒否 | 14 件拒否（httpx2 / httpcore2 2.16.0、truststore 0.10.5） |
| Python 3.11.18 / OpenAI 3.39.0 / certifi 2026.10.19 | 97 成功 | 14 件拒否 | 14 件拒否（httpx2 / httpcore2 2.16.0、truststore 0.10.5） |

すべての更新後検査で正常対照は成功。既定環境と各空環境で依存整合性を確認した。
更新前の受け入れ 21 接続は、公表されたリーフ・中間証明書の指紋と一致した。
更新後の拒否理由は OpenSSL の `certificate rejected`（28）、
`unable to get local issuer certificate`（20）、macOS の失効エラー（-67820）。
macOS のコードは公式 SDK の Security.framework `SecBase.h` で
`errSecCertificateRevoked` と確認した。期限切れや CT のみでの拒否には分類していない。

通常起動 `uv run python` が `.venv/bin/python` / Python 3.13.18 を使うことを確認した。
ベース Python は uv 標準の `~/.local/share/uv/python/` 配下で、`.tools/python/` には依存しない。
旧 `.venv` は `.venv-pre-emudhra-20261027/` に保持。復元用の退避なので、直接起動せず元のパスに
戻して使用する（旧スクリプトの絶対パスは元の `.venv` のまま）。
検証用に新規作成した一時環境だけは、既存の確認スクリプトの終了時に削除される。
当初作った専用ツール方式の `.venv` も `.venv-pre-emudhra-project-tools-20261027/` に保持した。
`.tools/` に残る以前の実行ファイルは使用しない。uv が当初作ったバージョン付き `python3.13` リンクは
uv 標準の保存先へ合わせた。汎用 `python` / `python3` の既定選択は変更していない。

### 未確認・制約

- CI・Docker 設定はリポジトリ内に存在せず、実行していない。
- uv 独自 TLS は更新版による PyPI・Python 配布元への通常接続を確認したが、
  リダイレクト禁止を指定できる検査手段を用意していないため、14 件の直接拒否は未確認。
- VS Code 1.143.0 のユーザー既定は `/usr/bin/python3`（3.9.6）だった。
  ワークスペース設定を `.venv` に追加したが、IDE 内で既に選択済みの実体は未確認。
- 実ゲートウェイの起動確認とモデル一覧取得（5 件）は成功した。
  README の既定モデルは未登録だが、取得済みの `ToPo-ToPo/Qwen3.8-27B-mlx-4bit` を指定し、
  通常応答「1 + 1」→「2」、ストリーミング「日本の首都」→「東京」を実モデルで確認した。
  所要時間は初回ロード込み 26.246 秒、後続ストリーミング 0.269 秒。サーバーログでも両要求の HTTP 200 を確認。
  Python 3.13.18 / OpenAI 2.44.0、通常の `uv run python` から実行した。追加ダウンロード・設定変更は行っていない。
  音声・画像・動画・ツール呼び出し等は既存テストで確認し、実モデルでの検証は未実施。
- Python 3.12 / 3.14 / 3.15 の実行、CPython 以外の実装、Windows/Linux は未確認。
  修正版の範囲は公式リリース情報に基づく。

詳細ログはこの作業環境の `.tools/validation/emudhra-20261027/` に保存（Git 管理外）。
主要ファイルは `baseline-environment.json`、`baseline-tls.json`、`updated-environment.json`、
`updated-tests.log`、`updated-tls.json`、`matrix-min-certifi.log`、
`matrix-min-certifi/*-tls.json`、`gateway-smoke.json`。
従来の手順へ戻した後の再確認ログは同フォルダの `traditional-uv/` に保存。
追加の実モデル生成結果は `live-generation.json`、サーバーログの該当部分は `live-generation-server.log` に保存。

### 公式情報

- [対象証明書と 14 接続先・チェーン資料](https://emudhra.we-have-a-problem.com/)
- [Mozilla のインシデント報告](https://bugzilla.mozilla.org/show_bug.cgi?id=2083449)
- [Apple の対応と標準 CA ファイルの注意点](https://support.apple.com/en-us/149555)
- [Python 各系列の ssl 修正](https://blog.python.org/2026/10/python-3152-31410-31318-31216-31118/)
- [Python 向け certifi / pip / uv の告知](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576/1)
- [uv 0.13.5 の追加対策](https://github.com/astral-sh/uv/releases/tag/0.13.5)
- [uv 0.13.6 の Python カタログ](https://github.com/astral-sh/uv/releases/tag/0.13.6)
- [OpenAI SDK の HTTPX2・証明書ストア変更](https://github.com/openai/openai-python/blob/main/httpx2.md)

### 0.10.3: 公開依存要件の補完（2026-10-28）

開発環境のロック更新だけでは、PyPI 利用者の既存環境に修正前の間接依存が残り得る。
0.10.3 では配布 wheel / sdist の `Requires-Dist` にも次の下限を追加した。
OpenAI 1〜3 系の対応は維持する。HTTP 層を SDK の版で条件分岐した依存宣言はできないため、
httpx / httpx2 の両方をインストールする。コードが使用するのは選択した SDK の既定 HTTP 層。

| 依存 | 公開要件の下限 | 開発環境のロック・実行版 | 理由 |
| --- | --- | --- | --- |
| OpenAI SDK | 1.55.3、上限 <4 | 2.44.0 | 従来の対応範囲を維持 |
| certifi | 2026.10.19 | 2026.10.19 | eMudhra の CA 除外 |
| AnyIO | 4.14.2、上限 <5 | 4.15.1 | TLS ホスト名検証などの修正 |
| httpx / httpcore | 0.28.1 / 1.0.9 | 0.28.1 / 1.0.9 | SDK 1・2 系と h11>=0.16 の framing 修正 |
| httpx2 | 2.12.0、上限 <3 | 2.16.0 | SDK 3 系の HTTP framing・解凍処理などの修正 |
| tqdm | 4.66.3、上限 <5 | 4.68.4 | 古い SDK が許可する脆弱版を除外 |
| Pydantic | 1.10.20、2.0〜2.3 は除外、上限 <3 | 2.13.4 | EmailStr の修正済み範囲（1.10.13 / 2.4.0 以降）。1 系の下限は Python 3.13 対応も含む |
| typing-extensions | 4.12.2、上限 <5 | 4.16.0 | 最小構成で確認した Python 3.13 の ParamSpec 非互換を回避 |
| setuptools（ビルド用） | 84.1.0 | wheel の Generator に記録 | ビルドと旧 wheel インストール処理の脆弱版を除外 |

Python の下限は引き続き系列ごとに 3.11.18 / 3.12.16 / 3.13.18 / 3.14.10 / 3.15.2。
`Requires-Python` の除外はパッチ系列の `!=3.13.17.*` などで記載する。
ただし uv 0.13.6 の依存解決は途中の Python 除外区間を無視するケースがあった。
そのため import の最初にも版を検査し、未修正 Python では HTTP SDK を読み込む前に
必要な更新版を表示して停止する。独自の起動スクリプトは導入しない。

検証では配布 wheel をソースツリーの外の空環境へ入れた。
Python 3.11.18 上の OpenAI 1.55.3 / 2.0.0 / 2.54.0 / 3.0.0 / 3.39.0 で各 102 テストが成功。
各環境で urllib / SDK の正常対照が成功し、それぞれ問題の 14/14 件を証明書検証で拒否した。
Python 3.13.18 の最小依存構成でも既存機能を検証し、開発環境では通常応答「2」と
ストリーミング応答「東京」を既存のローカル LLM から取得した。

実際の配布要件を使った依存解決では修正前の 11 構成が拒否された。
pip 26.2.2 で上記 5 系列の未修正 Python を拒否し、実際の各旧 Python から wheel を
読み込んだ場合も、通信開始前に停止することを確認した。
最小依存とロック環境を PyPI の脆弱性情報で照合し、該当・取得失敗とも 0 件。
この照合は実施時点の公開情報の範囲であり、将来のすべての脆弱性を保証するものではない。
CI・コンテナ・他 OS・独自 TLS の未確認範囲は前節のまま。

ログは `.tools/validation/dependency-security-20261028/` に保存する（配布物・Git の対象外）。
`client-matrix.log`、`client-min-tests.log`、`client-min-tls.json`、
`client-current-advisories.json`、`public-resolver-results.json` で依存の実測・判定理由を追える。

追加の根拠:

- [AnyIO の TLS 修正](https://github.com/agronholm/anyio/security/advisories/GHSA-82r6-8w77-94w6)
- [HTTPX2 の修正一覧](https://github.com/pydantic/httpx2/security/advisories)
- [setuptools 84.1.0 の修正](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w)
- [uv の Requires-Python 解決方針](https://docs.astral.sh/uv/reference/resolver-internals/#requires-python)
