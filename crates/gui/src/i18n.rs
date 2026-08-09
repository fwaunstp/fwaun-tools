//! GUI internationalization. The app supports English and Japanese; default
//! is the host locale. The user's choice persists in `gui-prefs.toml`
//! alongside the other machine-local settings — see [`crate::prefs`].

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
    pub fn reload_folder(self) -> &'static str {
        self.pair("Reload", "再読み込み")
    }
    pub fn reload_folder_title(self) -> &'static str {
        self.pair(
            "Re-scan the current folder. Thumbnails are regenerated only for \
             new or modified images; sidecars are re-read for all of them.",
            "現在のフォルダを再スキャンします。サムネイルは新規・変更された画像のみ\
             作り直し、サイドカーは全件読み直します。",
        )
    }
    pub fn config_button(self) -> &'static str {
        self.pair("Config…", "設定…")
    }
    pub fn config_button_title(self) -> &'static str {
        self.pair(
            "Edit fwaun-tools.toml for the current dataset folder",
            "現在のデータセットフォルダの fwaun-tools.toml を編集します",
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
            "Define [tag_group.<name>] in fwaun-tools.toml to enable Kanban views.",
            "カンバン表示を使うには fwaun-tools.toml で [tag_group.<name>] を定義してください。",
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
    pub fn op_loading_folder(self) -> &'static str {
        self.pair("Loading folder…", "フォルダ読み込み中…")
    }
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
    pub fn cancel(self) -> &'static str {
        self.pair("Cancel", "キャンセル")
    }
    pub fn cancelling(self) -> &'static str {
        self.pair("Cancelling…", "キャンセル中…")
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

    // Preview / open with the OS
    pub fn open_preview(self) -> &'static str {
        self.pair("Preview", "拡大表示")
    }
    pub fn open_preview_title(self) -> &'static str {
        self.pair(
            "Show this image full size (or double-click the thumbnail). \
             ←/→ move between images, Esc closes.",
            "この画像を原寸表示します（サムネイルのダブルクリックでも開きます）。\
             ←/→ で前後の画像へ、Esc で閉じます。",
        )
    }
    pub fn open_external(self) -> &'static str {
        self.pair("Open in default app", "既定のアプリで開く")
    }
    pub fn reveal_in_folder(self) -> &'static str {
        self.pair("Show in folder", "フォルダで表示")
    }
    pub fn close(self) -> &'static str {
        self.pair("Close", "閉じる")
    }
    pub fn preview_prev(self) -> &'static str {
        self.pair("Previous image (←)", "前の画像 (←)")
    }
    pub fn preview_next(self) -> &'static str {
        self.pair("Next image (→)", "次の画像 (→)")
    }
    pub fn preview_position(self, current: usize, total: usize) -> String {
        match self.lang {
            Lang::En => format!("{current} / {total}"),
            Lang::Ja => format!("{current} / {total} 件目"),
        }
    }
    pub fn preview_decode_failed(self, err: &str) -> String {
        match self.lang {
            Lang::En => format!("Could not load this image: {err}"),
            Lang::Ja => format!("この画像を読み込めませんでした: {err}"),
        }
    }
    pub fn err_open_external(self, path: &str, err: &str) -> String {
        match self.lang {
            Lang::En => format!("could not hand {path} to the OS: {err}"),
            Lang::Ja => format!("{path} をOSに渡せませんでした: {err}"),
        }
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
            "Caption hints (passed to captioner only)",
            "キャプションヒント（キャプショナーにのみ渡されます）",
        )
    }
    pub fn empty_hints(self) -> &'static str {
        self.pair(
            "(none yet — add a reference fact below)",
            "（まだありません — 下の欄に参考情報を追加してください）",
        )
    }
    pub fn add_hint_placeholder(self) -> &'static str {
        self.pair(
            "Add a reference fact, e.g. \"blue hair girl is Laundry Dragonmaid\" — added to every selected image, one bullet per fact. Sent to the captioner only.",
            "参考情報を追加（例: 「blue hair girl is Laundry Dragonmaid」）。選択中のすべての画像に1項目ずつ追加され、キャプショナーにのみ渡されます。",
        )
    }
    pub fn section_manual_caption(self) -> &'static str {
        self.pair(
            "Caption (manual — exported)",
            "キャプション（手動・書き出し対象）",
        )
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
        self.pair(
            "(none — run captioner)",
            "（なし — キャプショナーを実行してください）",
        )
    }
    pub fn manual_caption_placeholder(self) -> &'static str {
        self.pair(
            "Manual caption — exported verbatim, overrides any auto captions. Leave empty to export the auto captions instead. Click outside to save.",
            "手動キャプション — そのまま書き出され、自動キャプションを上書きします。空のままだと自動キャプションが書き出されます。フォーカスを外すと保存されます。",
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
            "Caption hints (add to all selected)",
            "キャプションヒント（選択中すべてに追加）",
        )
    }
    pub fn section_manual_entries(self) -> &'static str {
        self.pair("Manual entries (union)", "手動エントリ（和集合）")
    }
    /// Bulk rename row under the manual entries — the in-place counterpart
    /// to removing a chip and typing the tag back in.
    pub fn rename_from_placeholder(self) -> &'static str {
        self.pair("old tag", "旧タグ")
    }
    pub fn rename_to_placeholder(self) -> &'static str {
        self.pair("new tag", "新タグ")
    }
    pub fn rename_button(self) -> &'static str {
        self.pair("Rename", "リネーム")
    }
    pub fn rename_button_title(self) -> &'static str {
        self.pair(
            "Rename the manual entry across the selected images, keeping its position. \
             Images without the old tag are left alone.",
            "選択中の画像の手動エントリを、位置を保ったままリネームします。旧タグを持たない\
             画像は変更しません。",
        )
    }
    pub fn rename_tag_menu(self) -> &'static str {
        self.pair("Rename this tag…", "このタグをリネーム…")
    }
    pub fn rename_tag_menu_title(self) -> &'static str {
        self.pair(
            "Fill this tag into the rename row below.",
            "下のリネーム欄にこのタグを入れます。",
        )
    }
    pub fn section_shared_tags(self) -> &'static str {
        self.pair(
            "Shared by selection (auto/booru, ≥2 images)",
            "選択内で共通（自動/Booru、2件以上）",
        )
    }
    /// The dataset-wide `common_tags` layer from `fwaun-tools.toml` — not to
    /// be confused with [`section_shared_tags`], which counts tags the
    /// current selection happens to share.
    pub fn section_dataset_tags(self) -> &'static str {
        self.pair(
            "Dataset tags (fwaun-tools.toml)",
            "データセット共通タグ（fwaun-tools.toml）",
        )
    }
    pub fn dataset_tags_root_hint(self) -> &'static str {
        self.pair(
            "All images selected at the dataset root — adding or removing a tag edits fwaun-tools.toml, not the sidecars.",
            "データセットルートで全画像を選択中 — タグの追加・削除はサイドカーではなく fwaun-tools.toml を編集します。",
        )
    }
    pub fn dataset_tags_override_hint(self) -> &'static str {
        self.pair(
            "Click a tag to override it for the selected image(s) only.",
            "タグをクリックすると、選択中の画像だけで打ち消します。",
        )
    }
    pub fn common_tag_write_failed(self, path: &str, err: &str) -> String {
        match self.lang {
            Lang::En => format!("Failed to update common_tags in {path}: {err}"),
            Lang::Ja => format!("{path} の common_tags 更新に失敗: {err}"),
        }
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
    pub fn cfg_tab_app(self) -> &'static str {
        self.pair("App", "アプリ")
    }

    // App tab (machine-local prefs, `gui-prefs.toml`)
    pub fn cfg_app_note(self) -> &'static str {
        self.pair(
            "These settings belong to this machine, not to the dataset. They are saved to gui-prefs.toml as soon as you change them — Save / Cancel below only apply to the dataset config above.",
            "ここの設定はデータセットではなくこのマシンに紐づきます。変更すると即座に gui-prefs.toml へ保存されます（下の「保存」「キャンセル」は上のデータセット設定にのみ適用されます）。",
        )
    }
    pub fn cfg_thumb_cache(self) -> &'static str {
        self.pair("Thumbnail cache", "サムネイルキャッシュ")
    }
    pub fn cfg_thumb_cache_enabled(self) -> &'static str {
        self.pair("Cache thumbnails on disk", "サムネイルをディスクに保存する")
    }
    pub fn cfg_thumb_cache_help(self) -> &'static str {
        self.pair(
            "Keeps generated thumbnails under the OS cache directory so re-opening a folder skips decoding the full-size images. Entries are keyed by path + modification time, so an edited image regenerates automatically.",
            "生成済みサムネイルを OS のキャッシュディレクトリに保存し、フォルダを開き直したときに元画像のデコードを省きます。エントリはパスと更新時刻で識別されるため、画像を差し替えれば自動的に作り直されます。",
        )
    }
    pub fn cfg_thumb_cache_limit(self) -> &'static str {
        self.pair("Size limit (MiB)", "上限サイズ（MiB）")
    }
    pub fn cfg_thumb_cache_max_age(self) -> &'static str {
        self.pair("Expire after (days)", "有効期限（日）")
    }
    pub fn cfg_thumb_cache_zero_off(self) -> &'static str {
        self.pair("0 = no limit", "0 で無制限")
    }
    pub fn cfg_thumb_cache_size(self, size: &str) -> String {
        match self.lang {
            Lang::En => format!("Currently using {size}"),
            Lang::Ja => format!("現在の使用量: {size}"),
        }
    }
    pub fn cfg_thumb_cache_size_unknown(self) -> &'static str {
        self.pair("Currently using —", "現在の使用量: —")
    }
    pub fn cfg_thumb_cache_measure(self) -> &'static str {
        self.pair("Measure", "計測")
    }
    pub fn cfg_thumb_cache_clear(self) -> &'static str {
        self.pair("Clear cache", "キャッシュを削除")
    }
    pub fn cfg_thumb_cache_unavailable(self) -> &'static str {
        self.pair(
            "This platform reports no cache directory; thumbnails cannot be cached.",
            "このプラットフォームではキャッシュディレクトリを取得できないため、サムネイルを保存できません。",
        )
    }
    pub fn cfg_thumb_cache_clear_failed(self, err: &str) -> String {
        match self.lang {
            Lang::En => format!("Could not clear the thumbnail cache: {err}"),
            Lang::Ja => format!("サムネイルキャッシュを削除できませんでした: {err}"),
        }
    }
    pub fn cfg_default_profile(self) -> &'static str {
        self.pair("Default export profile", "既定エクスポートプロファイル")
    }
    pub fn cfg_default_tagger(self) -> &'static str {
        self.pair("Default tagger profile", "既定タガープロファイル")
    }
    pub fn cfg_default_captioner(self) -> &'static str {
        self.pair(
            "Default captioner profile",
            "既定キャプショナープロファイル",
        )
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
    pub fn cfg_common_tags(self) -> &'static str {
        self.pair(
            "Dataset tags (common_tags)",
            "データセット共通タグ（common_tags）",
        )
    }
    pub fn cfg_common_tags_help(self) -> &'static str {
        self.pair(
            "One per line, applied to every image without touching any sidecar: `foo` positive, `-foo` suppression, `_foo` curation-only. Typically the character trigger word plus suppressions for the traits it should absorb (hair/eye colour). An image opts out by setting the tag itself.",
            "1行に1エントリ。サイドカーを書き換えずに全画像へ適用されます（`foo` 通常、`-foo` 抑制、`_foo` 整理用）。通常はキャラ名のトリガーワードと、そこに吸収させたい特徴（髪色・目色など）の抑制を並べます。個別画像側で同じタグを指定すれば打ち消せます。",
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
        self.pair(
            "Max edge (resize, 0=off)",
            "最大辺サイズ（リサイズ、0で無効）",
        )
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
    pub fn cfg_empty_retries(self) -> &'static str {
        self.pair("Empty-caption retries", "空キャプション時リトライ回数")
    }
    pub fn cfg_empty_retries_hint(self) -> &'static str {
        self.pair(
            "How many times to regenerate when the caption comes back empty. After the retries are used up the empty result is reported as an error instead of being saved, so it will be retried on the next run.",
            "キャプションが空で返ってきたときに再生成する回数。リトライを使い切っても空の場合は保存せずエラーとして報告するため、次回実行時に再試行されます。",
        )
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
    pub fn cfg_caption_prefixes(self) -> &'static str {
        self.pair("Caption prefixes", "キャプションプレフィックス")
    }
    pub fn cfg_caption_suffixes(self) -> &'static str {
        self.pair("Caption suffixes", "キャプションサフィックス")
    }
    pub fn cfg_tag(self) -> &'static str {
        self.pair("tag", "タグ")
    }
    pub fn cfg_suffix(self) -> &'static str {
        self.pair("suffix", "サフィックス")
    }
    pub fn cfg_tags(self) -> &'static str {
        self.pair("Tags", "タグ")
    }
    pub fn cfg_tag_groups_note(self) -> &'static str {
        self.pair(
            "Tag groups define tag sets. Exclusive groups drive the Kanban view (one column per tag plus \"unset\"/\"violation\"). A group's caption hint / prefix / suffix apply when ALL its tags are present — set exclusive off for co-occurring steering tags.",
            "タググループはタグの集合を定義します。排他グループはカンバン表示に使われます（タグごとに 1 列 + 「未設定」「違反」）。キャプションのヒント／プレフィクス／サフィックスはグループの全タグが揃ったときに適用されます。共起させるステアリング用タグは排他をオフにしてください。",
        )
    }
    pub fn cfg_tag_group_exclusive(self) -> &'static str {
        self.pair(
            "Exclusive (mutually-exclusive tags)",
            "排他（相互排他タグ）",
        )
    }
    pub fn cfg_tag_group_caption_hint(self) -> &'static str {
        self.pair(
            "Caption hint (fed to the model)",
            "キャプションヒント（モデルに渡す）",
        )
    }
    pub fn cfg_tag_group_caption_prefix(self) -> &'static str {
        self.pair("Caption prefix", "キャプションプレフィクス")
    }
    pub fn cfg_tag_group_caption_suffix(self) -> &'static str {
        self.pair("Caption suffix", "キャプションサフィックス")
    }
    pub fn cfg_tag_group_priority(self) -> &'static str {
        self.pair("priority", "優先度")
    }
    pub fn cfg_tag_group_affix_note(self) -> &'static str {
        self.pair(
            "Prefix/suffix are folded into the exported caption and steer generation; ascending priority orders concatenation when several groups match. Groups sharing a priority are ordered per image (same order on every run) so the dataset isn't trained on one fixed sequence. Leave content empty to disable.",
            "プレフィクス／サフィックスはエクスポート時のキャプションに畳み込まれ、生成も誘導します。複数グループが一致した場合は優先度の昇順で連結されます。優先度が同じグループ同士は画像ごとに順番が入れ替わります（同じ画像なら毎回同じ順番）。順番の固定を学習させないための挙動です。内容を空にすると無効になります。",
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
            Lang::En => {
                format!("Existing config could not be parsed; starting from defaults: {err}")
            }
            Lang::Ja => format!("既存の設定を解析できなかったため、既定値から編集します: {err}"),
        }
    }

    // Errors raised from the UI (most others come from anyhow / external).
    pub fn err_open_folder_first(self) -> String {
        self.pair("Open a folder first.", "先にフォルダを開いてください。")
            .to_string()
    }

    // ───────── Top-level mode tabs ─────────
    pub fn mode_dataset(self) -> &'static str {
        self.pair("Dataset", "データセット")
    }
    pub fn mode_model(self) -> &'static str {
        self.pair("Model tools", "モデルツール")
    }

    // ───────── Model tools tab ─────────
    pub fn model_op_merge(self) -> &'static str {
        self.pair("Merge diff", "差分マージ")
    }
    pub fn model_op_extract(self) -> &'static str {
        self.pair("Extract LoRA", "LoRA抽出")
    }
    pub fn model_op_quant(self) -> &'static str {
        self.pair("Quantize int8", "int8量子化")
    }
    pub fn model_op_merge_desc(self) -> &'static str {
        self.pair(
            "output = target + multiplier × (tuned − base). Transfers a full \
             fine-tune delta onto another checkpoint.",
            "output = target + multiplier ×（tuned − base）。フル微調整の差分を別の\
             チェックポイントへ転写します。",
        )
    }
    pub fn model_op_extract_desc(self) -> &'static str {
        self.pair(
            "SVD of (tuned − base) into a low-rank kohya/ComfyUI LoRA.",
            "（tuned − base）をSVDして低ランクの kohya/ComfyUI LoRA にします。",
        )
    }
    pub fn model_op_quant_desc(self) -> &'static str {
        self.pair(
            "Quantize a bf16/fp16 checkpoint to int8 + ConvRot (comfy-kitchen layout).",
            "bf16/fp16 チェックポイントを int8 + ConvRot（comfy-kitchen 形式）へ量子化します。",
        )
    }
    // Field labels
    pub fn model_field_base(self) -> &'static str {
        self.pair("Base (original)", "ベース（元モデル）")
    }
    pub fn model_field_tuned(self) -> &'static str {
        self.pair("Tuned (fine-tune)", "微調整後モデル")
    }
    pub fn model_field_target(self) -> &'static str {
        self.pair("Target (receiver)", "ターゲット（適用先）")
    }
    pub fn model_field_output(self) -> &'static str {
        self.pair("Output", "出力先")
    }
    pub fn model_field_src(self) -> &'static str {
        self.pair("Source checkpoint", "入力チェックポイント")
    }
    pub fn model_field_dst(self) -> &'static str {
        self.pair("Output (optional)", "出力先（任意）")
    }
    pub fn model_field_multiplier(self) -> &'static str {
        self.pair("Multiplier", "係数")
    }
    pub fn model_field_arch(self) -> &'static str {
        self.pair("Key convention", "キー規約")
    }
    pub fn model_field_save_dtype(self) -> &'static str {
        self.pair("Save dtype", "保存dtype")
    }
    pub fn model_field_rank(self) -> &'static str {
        self.pair("Rank", "ランク")
    }
    pub fn model_field_alpha(self) -> &'static str {
        self.pair("Alpha (override)", "Alpha（上書き）")
    }
    pub fn model_field_include(self) -> &'static str {
        self.pair("Include regex (optional)", "含める正規表現（任意）")
    }
    pub fn model_field_exclude(self) -> &'static str {
        self.pair("Exclude regex (optional)", "除外する正規表現（任意）")
    }
    pub fn model_field_dry_run(self) -> &'static str {
        self.pair("Dry run (report only)", "ドライラン（計画のみ）")
    }
    pub fn model_field_min_gemm(self) -> &'static str {
        self.pair("Min GEMM", "最小GEMM")
    }
    pub fn model_field_warn_thresh(self) -> &'static str {
        self.pair("Warn threshold (relerr %)", "警告しきい値（相対誤差 %）")
    }
    pub fn model_dtype_keep(self) -> &'static str {
        self.pair("keep target's", "ターゲットのまま")
    }
    pub fn model_browse(self) -> &'static str {
        self.pair("Browse…", "参照…")
    }
    pub fn model_run(self) -> &'static str {
        self.pair("Run", "実行")
    }
    pub fn model_running(self) -> &'static str {
        self.pair(
            "Running… (this can take a while)",
            "実行中…（時間がかかることがあります）",
        )
    }
    pub fn model_err_need_paths(self) -> String {
        self.pair(
            "Fill in all required file paths first.",
            "先に必須のファイルパスをすべて入力してください。",
        )
        .to_string()
    }
    pub fn model_log_start(self, op: &str) -> String {
        match self.lang {
            Lang::En => format!("▶ {op} started…"),
            Lang::Ja => format!("▶ {op} を開始しました…"),
        }
    }
    pub fn model_log_ok(self, op: &str) -> String {
        match self.lang {
            Lang::En => format!("✔ {op} finished."),
            Lang::Ja => format!("✔ {op} が完了しました。"),
        }
    }
    pub fn model_log_err(self, op: &str, err: &str) -> String {
        match self.lang {
            Lang::En => format!("✖ {op} failed: {err}"),
            Lang::Ja => format!("✖ {op} が失敗しました: {err}"),
        }
    }
}
