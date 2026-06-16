//! GUI internationalization. The app supports English and Japanese; default
//! is the host locale. The user's choice persists in
//! `$XDG_CONFIG_HOME/anima-tagger/gui-prefs.toml` (or the `$HOME/.config`
//! fallback).

use std::fs;
use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    En,
    Ja,
}

impl Lang {
    pub fn detect_host() -> Self {
        sys_locale::get_locale()
            .map(|s| Self::from_locale_str(&s))
            .unwrap_or(Lang::En)
    }

    pub fn from_locale_str(s: &str) -> Self {
        if s.to_lowercase().starts_with("ja") {
            Lang::Ja
        } else {
            Lang::En
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Ja => "ja",
        }
    }
}

#[derive(Clone, Copy)]
pub struct T {
    pub lang: Lang,
}

impl T {
    pub fn new(lang: Lang) -> Self {
        Self { lang }
    }

    fn pair(self, en: &'static str, ja: &'static str) -> &'static str {
        match self.lang {
            Lang::En => en,
            Lang::Ja => ja,
        }
    }

    // Toolbar
    pub fn open_folder(self) -> &'static str {
        self.pair("Open folder…", "フォルダを開く…")
    }
    pub fn config_button(self) -> &'static str {
        self.pair("Config…", "設定…")
    }
    pub fn config_button_title(self) -> &'static str {
        self.pair(
            "Edit anima-tagger.toml for the current dataset folder",
            "現在のデータセットフォルダの anima-tagger.toml を編集します",
        )
    }
    pub fn no_folder(self) -> &'static str {
        self.pair("(no folder)", "（フォルダ未選択）")
    }
    pub fn filter_all(self) -> &'static str {
        self.pair("All", "すべて")
    }
    pub fn filter_untagged(self) -> &'static str {
        self.pair("Untagged", "未タグ付け")
    }
    pub fn filter_auto_tagged(self) -> &'static str {
        self.pair("Auto-tagged", "自動タグ付け済")
    }
    pub fn filter_no_manual(self) -> &'static str {
        self.pair("No manual tags", "手動タグなし")
    }
    pub fn filter_no_caption(self) -> &'static str {
        self.pair("No caption", "キャプションなし")
    }
    pub fn filter_no_hint(self) -> &'static str {
        self.pair("No caption hint", "キャプションヒントなし")
    }
    pub fn filter_no_booru(self) -> &'static str {
        self.pair("No booru", "Booru未取得")
    }
    pub fn tag_filter_placeholder(self) -> &'static str {
        self.pair("filter by tag…", "タグで絞り込み…")
    }
    pub fn select_visible(self) -> &'static str {
        self.pair("Select visible", "表示中を選択")
    }
    pub fn clear_selection(self) -> &'static str {
        self.pair("Clear sel.", "選択解除")
    }
    pub fn run_tagger(self) -> &'static str {
        self.pair("Run tagger", "タガーを実行")
    }
    pub fn run_captioner(self) -> &'static str {
        self.pair("Run captioner", "キャプショナーを実行")
    }
    pub fn fetch_booru(self) -> &'static str {
        self.pair("Fetch booru", "Booru取得")
    }
    pub fn working(self) -> &'static str {
        self.pair("Working…", "処理中…")
    }
    pub fn images_selected_summary(self, count: usize, sel_count: usize) -> String {
        match self.lang {
            Lang::En => format!("{count} images · {sel_count} selected"),
            Lang::Ja => format!("{count} 件 ・ {sel_count} 件選択中"),
        }
    }

    // View / Kanban
    pub fn view_grid(self) -> &'static str {
        self.pair("Grid", "グリッド")
    }
    pub fn view_kanban_prefix(self) -> &'static str {
        self.pair("Kanban: ", "カンバン: ")
    }
    pub fn kanban_no_groups_hint(self) -> &'static str {
        self.pair(
            "Define [tag_group.<name>] in anima-tagger.toml to enable Kanban views.",
            "カンバン表示を使うには anima-tagger.toml で [tag_group.<name>] を定義してください。",
        )
    }
    pub fn kanban_unset_column(self) -> &'static str {
        self.pair("(unset)", "（未設定）")
    }
    pub fn kanban_violation_column(self) -> &'static str {
        self.pair("(violation)", "（違反）")
    }
    pub fn kanban_drop_failed(self, path: &str, err: &str) -> String {
        match self.lang {
            Lang::En => format!("Failed to save sidecar for {path}: {err}"),
            Lang::Ja => format!("{path} のサイドカー保存に失敗: {err}"),
        }
    }

    // Progress overlay
    pub fn op_tagging(self) -> &'static str {
        self.pair("Tagging…", "タグ付け中…")
    }
    pub fn op_captioning(self) -> &'static str {
        self.pair("Captioning…", "キャプション生成中…")
    }
    pub fn op_fetching_booru(self) -> &'static str {
        self.pair("Fetching booru…", "Booru取得中…")
    }
    pub fn progress_count(self, current: usize, total: usize) -> String {
        match self.lang {
            Lang::En => format!("{current} / {total} images"),
            Lang::Ja => format!("{current} / {total} 件"),
        }
    }

    // Grid / status flags
    pub fn no_images(self) -> &'static str {
        self.pair("No images.", "画像がありません。")
    }
    pub fn thumb_status_title(self) -> &'static str {
        self.pair(
            "T=auto-tagged · C=captioned · B=booru · M=manual tags · H=caption hint",
            "T=自動タグ ・ C=キャプション ・ B=Booru ・ M=手動タグ ・ H=キャプションヒント",
        )
    }

    // Detail panel — empty
    pub fn select_to_edit(self) -> &'static str {
        self.pair(
            "Select one or more images to edit tags.",
            "タグを編集するには画像を1枚以上選択してください。",
        )
    }
    pub fn tip_suppress(self) -> &'static str {
        self.pair(
            "Tip: \"-tag\" suppresses an auto/booru tag; \"_tag\" is a curation-only label (kept for tag-group sorting but never exported). Both stay in the data, out of the caption.",
            "ヒント: \"-タグ\" は自動/Booruタグの非表示指定、\"_タグ\" は整理用ラベル（タググループ分類には使われますが書き出されません）。どちらもデータには残り、キャプションには出ません。",
        )
    }

    // Detail panel — single
    pub fn section_tags(self) -> &'static str {
        self.pair("Tags", "タグ")
    }
    pub fn section_caption_hint(self) -> &'static str {
        self.pair(
            "Caption hint (passed to captioner only)",
            "キャプションヒント（キャプショナーにのみ渡されます）",
        )
    }
    pub fn section_manual_caption(self) -> &'static str {
        self.pair("Caption (manual — exported)", "キャプション（手動・書き出し対象）")
    }
    pub fn section_auto_captions(self) -> &'static str {
        self.pair("Auto captions", "自動キャプション")
    }
    pub fn section_booru(self) -> &'static str {
        self.pair("Booru", "Booru")
    }
    pub fn empty_tags(self) -> &'static str {
        self.pair(
            "(none yet — add manual or run tagger/booru)",
            "（まだありません — 手動追加するかタガー/Booruを実行してください）",
        )
    }
    pub fn empty_auto_captions(self) -> &'static str {
        self.pair("(none — run captioner)", "（なし — キャプショナーを実行してください）")
    }
    pub fn manual_caption_placeholder(self) -> &'static str {
        self.pair(
            "Manual caption — exported verbatim, overrides any auto captions. Leave empty to export the auto captions instead. Click outside to save.",
            "手動キャプション — そのまま書き出され、自動キャプションを上書きします。空のままだと自動キャプションが書き出されます。フォーカスを外すと保存されます。",
        )
    }
    pub fn caption_hint_placeholder(self) -> &'static str {
        self.pair(
            "Reference info for the captioner (e.g. \"The girl with red hair on the left is Alice; the boy on the right is Bob.\"). Sent as a system turn — does NOT appear in the exported .txt. Click outside to save.",
            "キャプショナー向けの参考情報（例: 「左の赤髪の女の子はAlice、右の男の子はBob」）。システムターンとして送られ、書き出される .txt には含まれません。フォーカスを外すと保存されます。",
        )
    }
    pub fn promote_to_manual(self) -> &'static str {
        self.pair("→ manual", "→ 手動へ")
    }
    pub fn promote_to_manual_title(self) -> &'static str {
        self.pair(
            "Copy this caption into the manual caption field",
            "このキャプションを手動キャプション欄にコピー",
        )
    }
    pub fn skip(self) -> &'static str {
        self.pair("skip", "除外")
    }
    pub fn unskip(self) -> &'static str {
        self.pair("unskip", "除外解除")
    }
    pub fn skip_title(self) -> &'static str {
        self.pair(
            "Keep this caption stored but exclude from export",
            "このキャプションは保持しつつ書き出しからは除外します",
        )
    }
    pub fn unskip_title(self) -> &'static str {
        self.pair(
            "Re-enable this caption for export",
            "このキャプションを書き出し対象に戻します",
        )
    }
    pub fn remove_caption_title(self) -> &'static str {
        self.pair("Remove this auto caption", "この自動キャプションを削除")
    }

    // Detail panel — bulk
    pub fn n_selected_bulk(self, n: usize) -> String {
        match self.lang {
            Lang::En => format!("{n} images selected — bulk edit"),
            Lang::Ja => format!("{n} 件選択中 — 一括編集"),
        }
    }
    pub fn section_bulk_caption_hint(self) -> &'static str {
        self.pair(
            "Caption hint (apply to all selected)",
            "キャプションヒント（選択中すべてに適用）",
        )
    }
    pub fn bulk_hints_differ(self) -> &'static str {
        self.pair(
            "(selected images have differing hints — applying will overwrite all)",
            "（選択中の画像でヒントが異なります — 適用するとすべて上書きされます）",
        )
    }
    pub fn bulk_hint_placeholder(self) -> &'static str {
        self.pair(
            "Reference info applied to every selected image. Sent to the captioner as a system turn.",
            "選択中のすべての画像に適用される参考情報。キャプショナーへシステムターンとして渡されます。",
        )
    }
    pub fn bulk_hint_apply(self) -> &'static str {
        self.pair("Apply to all selected", "選択中すべてに適用")
    }
    pub fn bulk_hint_clear(self) -> &'static str {
        self.pair("Clear", "クリア")
    }
    pub fn section_manual_entries(self) -> &'static str {
        self.pair("Manual entries (union)", "手動エントリ（和集合）")
    }
    pub fn section_common_tags(self) -> &'static str {
        self.pair("Common tags (auto/booru, ≥2 images)", "共通タグ（自動/Booru、2件以上）")
    }
    pub fn section_bulk_manual_caption(self) -> &'static str {
        self.pair("Manual caption (bulk)", "手動キャプション（一括）")
    }
    pub fn bulk_clear_manual(self) -> &'static str {
        self.pair("Clear manual", "手動キャプションをクリア")
    }
    pub fn bulk_clear_manual_title(self) -> &'static str {
        self.pair(
            "Clear manual_caption on all selected images so a follow-up promote can repopulate it.",
            "選択中すべての画像の手動キャプションをクリアします（その後 → 手動へ で再投入できます）。",
        )
    }
    pub fn section_bulk_auto_captions(self) -> &'static str {
        self.pair("Auto captions (by model)", "自動キャプション（モデル別）")
    }
    pub fn empty_simple(self) -> &'static str {
        self.pair("(none)", "（なし）")
    }
    pub fn bulk_promote_title(self) -> &'static str {
        self.pair(
            "Copy this caption into manual_caption on every selected image whose manual is empty.",
            "選択中で手動キャプションが空の画像すべてに、このキャプションをコピーします。",
        )
    }
    pub fn bulk_remove_caption_title(self) -> &'static str {
        self.pair(
            "Remove this model's caption from all selected",
            "このモデルのキャプションを選択中すべてから削除",
        )
    }
    pub fn switch_to_single_hint(self) -> &'static str {
        self.pair(
            "Switch to single selection to suppress individual auto/booru tags.",
            "個別の自動/Booruタグを非表示にするには、1枚だけ選択してください。",
        )
    }

    // Add input
    pub fn add_input_placeholder(self) -> &'static str {
        self.pair(
            "tag, -tag to suppress, _tag for curation-only",
            "タグ（-タグ で非表示、_タグ で整理用）",
        )
    }
    pub fn add_button(self) -> &'static str {
        self.pair("Add", "追加")
    }

    // Delete image
    pub fn delete_image(self) -> &'static str {
        self.pair("Delete image…", "画像を削除…")
    }
    pub fn delete_images(self) -> &'static str {
        self.pair("Delete selected images…", "選択中の画像を削除…")
    }
    pub fn delete_image_title(self) -> &'static str {
        self.pair(
            "Permanently delete the image file and its sidecar from disk.",
            "画像ファイルとサイドカーをディスクから完全に削除します。",
        )
    }
    pub fn delete_confirm_title(self) -> &'static str {
        self.pair("Delete image(s)", "画像を削除")
    }
    pub fn delete_confirm_body(self, n: usize) -> String {
        match self.lang {
            Lang::En => format!(
                "Permanently delete {n} image file(s) and their sidecars? This cannot be undone."
            ),
            Lang::Ja => format!(
                "{n} 件の画像ファイルとサイドカーを完全に削除します。元に戻せません。よろしいですか？"
            ),
        }
    }
    pub fn delete_confirm_ok(self) -> &'static str {
        self.pair("Delete", "削除")
    }
    pub fn delete_confirm_cancel(self) -> &'static str {
        self.pair("Cancel", "キャンセル")
    }
    pub fn err_delete_failed(self, path: &str, err: &str) -> String {
        match self.lang {
            Lang::En => format!("delete failed: {path}: {err}"),
            Lang::Ja => format!("削除に失敗しました: {path}: {err}"),
        }
    }

    // Tagger skip
    pub fn info_all_already_tagged(self) -> &'static str {
        self.pair(
            "All selected images are already auto-tagged. Nothing to do.",
            "選択中の画像はすべて自動タグ付け済みです。実行する処理はありません。",
        )
    }
    pub fn info_skipped_already_tagged(self, skipped: usize) -> String {
        match self.lang {
            Lang::En => format!("{skipped} already auto-tagged image(s) skipped."),
            Lang::Ja => format!("自動タグ付け済みの {skipped} 件をスキップしました。"),
        }
    }

    // Config modal
    pub fn config_save(self) -> &'static str {
        self.pair("Save & reload", "保存して再読み込み")
    }
    pub fn config_cancel(self) -> &'static str {
        self.pair("Cancel", "キャンセル")
    }
    pub fn cfg_window_title(self) -> &'static str {
        self.pair("Settings", "設定")
    }
    pub fn cfg_tab_general(self) -> &'static str {
        self.pair("General", "一般")
    }
    pub fn cfg_tab_tagger(self) -> &'static str {
        self.pair("Tagger", "タガー")
    }
    pub fn cfg_tab_captioner(self) -> &'static str {
        self.pair("Captioner", "キャプショナー")
    }
    pub fn cfg_tab_prompts(self) -> &'static str {
        self.pair("Prompts", "プロンプト")
    }
    pub fn cfg_tab_export(self) -> &'static str {
        self.pair("Export", "エクスポート")
    }
    pub fn cfg_tab_tag_groups(self) -> &'static str {
        self.pair("Tag groups", "タググループ")
    }
    pub fn cfg_default_profile(self) -> &'static str {
        self.pair("Default export profile", "既定エクスポートプロファイル")
    }
    pub fn cfg_default_tagger(self) -> &'static str {
        self.pair("Default tagger profile", "既定タガープロファイル")
    }
    pub fn cfg_default_captioner(self) -> &'static str {
        self.pair("Default captioner profile", "既定キャプショナープロファイル")
    }
    pub fn cfg_none(self) -> &'static str {
        self.pair("(none — use built-in)", "（未指定 — 組込みを使用）")
    }
    pub fn cfg_general_note(self) -> &'static str {
        self.pair(
            "Defaults are picked when no `--profile` / `--tagger` / `--captioner` is passed. Leaving these unset falls back to the built-in models.",
            "`--profile` / `--tagger` / `--captioner` を指定しなかった場合に使用される既定値です。未指定なら組込みモデルが使われます。",
        )
    }
    pub fn cfg_unnamed(self) -> &'static str {
        self.pair("unnamed", "名称未設定")
    }
    pub fn cfg_name(self) -> &'static str {
        self.pair("Name", "名前")
    }
    pub fn cfg_repo(self) -> &'static str {
        self.pair("HuggingFace repo", "HuggingFace リポジトリ")
    }
    pub fn cfg_revision(self) -> &'static str {
        self.pair("Revision (optional)", "リビジョン（任意）")
    }
    pub fn cfg_subdir(self) -> &'static str {
        self.pair("Subdirectory (optional)", "サブディレクトリ（任意）")
    }
    pub fn cfg_input_size(self) -> &'static str {
        self.pair("Input size (px)", "入力サイズ (px)")
    }
    pub fn cfg_storage_threshold(self) -> &'static str {
        self.pair("Storage threshold", "保存しきい値")
    }
    pub fn cfg_kind(self) -> &'static str {
        self.pair("Kind", "種別")
    }
    pub fn cfg_endpoint(self) -> &'static str {
        self.pair("Endpoint", "エンドポイント")
    }
    pub fn cfg_model(self) -> &'static str {
        self.pair("Model name (optional)", "モデル名（任意）")
    }
    pub fn cfg_api_key(self) -> &'static str {
        self.pair("API key (optional)", "APIキー（任意）")
    }
    pub fn cfg_max_pixels(self) -> &'static str {
        self.pair("Max pixels", "最大ピクセル数")
    }
    pub fn cfg_max_new_tokens(self) -> &'static str {
        self.pair("Max new tokens", "最大新規トークン数")
    }
    pub fn cfg_max_tokens(self) -> &'static str {
        self.pair("Max tokens", "最大トークン数")
    }
    pub fn cfg_temperature(self) -> &'static str {
        self.pair("Temperature (optional)", "温度（任意）")
    }
    pub fn cfg_max_edge(self) -> &'static str {
        self.pair("Max edge (resize, 0=off)", "最大辺サイズ（リサイズ、0で無効）")
    }
    pub fn cfg_jpeg_quality(self) -> &'static str {
        self.pair("JPEG quality", "JPEG品質")
    }
    pub fn cfg_timeout_secs(self) -> &'static str {
        self.pair("Timeout (sec)", "タイムアウト（秒）")
    }
    pub fn cfg_max_retries(self) -> &'static str {
        self.pair("Max retries", "最大リトライ回数")
    }
    pub fn cfg_prompts(self) -> &'static str {
        self.pair("Prompts", "プロンプト")
    }
    pub fn cfg_prompts_note(self) -> &'static str {
        self.pair(
            "Prompts are referenced by name from each captioner profile's `prompts = [...]`. The built-in `default` is always available; redefining `default` here overrides it.",
            "ここで定義したプロンプトは各キャプショナープロファイルの `prompts = [...]` から名前で参照されます。組込みの `default` は常に利用可能で、ここで `default` を定義すると上書きされます。",
        )
    }
    pub fn cfg_threshold(self) -> &'static str {
        self.pair("Threshold", "しきい値")
    }
    pub fn cfg_shuffle(self) -> &'static str {
        self.pair("Shuffle on export", "書き出し時にシャッフル")
    }
    pub fn cfg_exclude_categories(self) -> &'static str {
        self.pair("Exclude categories", "除外カテゴリ")
    }
    pub fn cfg_category_prefixes(self) -> &'static str {
        self.pair("Category prefixes", "カテゴリ別プレフィックス")
    }
    pub fn cfg_category(self) -> &'static str {
        self.pair("category", "カテゴリ")
    }
    pub fn cfg_prefix(self) -> &'static str {
        self.pair("prefix", "プレフィックス")
    }
    pub fn cfg_tags(self) -> &'static str {
        self.pair("Tags", "タグ")
    }
    pub fn cfg_tag_groups_note(self) -> &'static str {
        self.pair(
            "Tag groups define mutually-exclusive tag sets. The Kanban view shows one column per tag plus an \"unset\" and \"violation\" column.",
            "タググループは相互排他なタグの集合を定義します。カンバン表示ではタグごとに 1 列 + 「未設定」「違反」の列が表示されます。",
        )
    }
    pub fn cfg_add(self) -> &'static str {
        self.pair("+ Add", "+ 追加")
    }
    pub fn cfg_remove(self) -> &'static str {
        self.pair("Remove", "削除")
    }
    pub fn cfg_add_tagger(self) -> &'static str {
        self.pair("+ Add tagger profile", "+ タガープロファイルを追加")
    }
    pub fn cfg_add_captioner_onnx(self) -> &'static str {
        self.pair("+ Add ONNX captioner", "+ ONNX キャプショナーを追加")
    }
    pub fn cfg_add_captioner_openai(self) -> &'static str {
        self.pair("+ Add OpenAI captioner", "+ OpenAI キャプショナーを追加")
    }
    pub fn cfg_add_prompt(self) -> &'static str {
        self.pair("+ Add prompt", "+ プロンプトを追加")
    }
    pub fn cfg_add_export(self) -> &'static str {
        self.pair("+ Add export profile", "+ エクスポートプロファイルを追加")
    }
    pub fn cfg_add_tag_group(self) -> &'static str {
        self.pair("+ Add tag group", "+ タググループを追加")
    }
    pub fn cfg_err_empty_name(self, section: &str) -> String {
        match self.lang {
            Lang::En => format!("[{section}] entry has an empty name"),
            Lang::Ja => format!("[{section}] に名前が空のエントリがあります"),
        }
    }
    pub fn cfg_err_duplicate_name(self, section: &str, name: &str) -> String {
        match self.lang {
            Lang::En => format!("[{section}] has duplicate name `{name}`"),
            Lang::Ja => format!("[{section}] に重複した名前 `{name}` があります"),
        }
    }
    pub fn cfg_err_load(self, err: &str) -> String {
        match self.lang {
            Lang::En => format!(
                "Existing config could not be parsed; starting from defaults: {err}"
            ),
            Lang::Ja => format!(
                "既存の設定を解析できなかったため、既定値から編集します: {err}"
            ),
        }
    }

    // Errors raised from the UI (most others come from anyhow / external).
    pub fn err_open_folder_first(self) -> String {
        self.pair("Open a folder first.", "先にフォルダを開いてください。").to_string()
    }
}

// ───────── persistence ─────────

fn prefs_path() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME").filter(|s| !s.is_empty()) {
        return Some(
            PathBuf::from(xdg)
                .join("anima-tagger")
                .join("gui-prefs.toml"),
        );
    }
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(|home| {
            PathBuf::from(home)
                .join(".config")
                .join("anima-tagger")
                .join("gui-prefs.toml")
        })
}

pub fn load_pref_or_detect() -> Lang {
    let stored = prefs_path()
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| toml::from_str::<toml::Value>(&s).ok())
        .and_then(|v| {
            v.get("language")
                .and_then(|l| l.as_str())
                .map(|s| s.to_string())
        });
    match stored.as_deref() {
        Some("ja") => Lang::Ja,
        Some("en") => Lang::En,
        _ => Lang::detect_host(),
    }
}

pub fn save_pref(lang: Lang) {
    let Some(path) = prefs_path() else { return };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let body = format!("language = \"{}\"\n", lang.code());
    let _ = fs::write(&path, body);
}
