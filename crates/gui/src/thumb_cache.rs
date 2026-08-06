//! On-disk thumbnail cache.
//!
//! Opening a folder decodes every image at full resolution just to shrink it
//! to `THUMB_SIZE` — for a few thousand 4K PNGs that is the entire cost of the
//! operation, and it was paid again on every launch. This module persists the
//! *already shrunk* RGBA image as a small PNG under the platform cache
//! directory, so the second open of a dataset decodes a few KB per image
//! instead of a few MB.
//!
//! Invalidation is implicit. The cache key hashes the source path together
//! with its mtime and size ([`FileStamp`]) and the thumbnail resolution, so an
//! edited, replaced, or re-sized source simply misses and gets regenerated;
//! nothing has to detect the change or evict the old entry. The stale entry is
//! left for [`ThumbCache::prune`] to reclaim.
//!
//! Everything here is best-effort: an unwritable cache directory, a truncated
//! entry from a killed process, a hash collision at 128 bits — all degrade to
//! "decode the original", never to an error the user has to act on. The one
//! case worth naming is a collision, which would show the *wrong* thumbnail;
//! at 128 bits that is not a risk worth trading disk space or CPU against.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use egui::ColorImage;
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};

/// mtime + size of a source image.
///
/// Used for two independent jobs that happen to want the same two numbers:
/// deciding whether an already-loaded thumbnail is still current during a
/// differential reload, and keying the on-disk cache. Content hashing would be
/// stricter but means reading every byte of every image — exactly the cost
/// this is meant to avoid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStamp {
    /// Nanoseconds since the Unix epoch. `0` when the platform has no mtime
    /// or it predates the epoch — degrades to size-only comparison.
    pub mtime_ns: u128,
    pub size: u64,
}

impl FileStamp {
    pub fn of(path: &Path) -> Option<Self> {
        let md = fs::metadata(path).ok()?;
        let mtime_ns = md
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Some(Self {
            mtime_ns,
            size: md.len(),
        })
    }
}

/// A resolved cache slot: the 128-bit digest of (path, stamp, thumb size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheKey([u8; 16]);

impl CacheKey {
    pub fn new(path: &Path, stamp: FileStamp, thumb_size: u32) -> Self {
        // FNV-1a, hand-rolled rather than `DefaultHasher`, because the digest
        // ends up in a filename that has to keep resolving across builds —
        // `DefaultHasher`'s output is explicitly not stable between Rust
        // releases, which would silently wipe the cache on every toolchain
        // bump. Two independent 64-bit lanes (different offset bases) give
        // 128 bits without a real wide-hash implementation.
        let mut lo = FNV_OFFSET_A;
        let mut hi = FNV_OFFSET_B;
        let mut feed = |bytes: &[u8]| {
            for b in bytes {
                lo = (lo ^ (*b as u64)).wrapping_mul(FNV_PRIME);
                hi = (hi ^ (*b as u64)).wrapping_mul(FNV_PRIME);
            }
        };
        // `to_string_lossy` is fine as a key: a path with unpaired surrogates
        // hashes to the same slot as its replacement-char form, which at worst
        // costs a regeneration for a path that cannot round-trip anyway.
        feed(path.to_string_lossy().as_bytes());
        feed(b"\0");
        feed(&stamp.mtime_ns.to_le_bytes());
        feed(&stamp.size.to_le_bytes());
        feed(&thumb_size.to_le_bytes());
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&lo.to_le_bytes());
        out[8..].copy_from_slice(&hi.to_le_bytes());
        Self(out)
    }

    fn hex(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(32);
        for b in &self.0 {
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

const FNV_OFFSET_A: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_OFFSET_B: u64 = 0x9e37_79b9_7f4a_7c15;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Handle to the cache directory. Cheap to clone — it's a path — so the
/// folder-scan worker thread gets its own copy.
#[derive(Debug, Clone)]
pub struct ThumbCache {
    root: PathBuf,
}

impl ThumbCache {
    /// `None` when the platform has no cache directory. Does not create
    /// anything; the directory appears on the first successful [`Self::store`].
    pub fn open() -> Option<Self> {
        dirs::cache_dir().map(|d| Self {
            root: d.join("fwaun-tools").join("thumbnails"),
        })
    }

    /// A cache rooted at an arbitrary directory, so tests don't write into
    /// the real user cache.
    #[cfg(test)]
    pub fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Entries are sharded by the first hex byte of the digest. 256 buckets
    /// keeps directory listings small enough that a prune over a cache with
    /// tens of thousands of entries stays quick on every filesystem.
    fn entry_path(&self, key: &CacheKey) -> PathBuf {
        let hex = key.hex();
        self.root.join(&hex[..2]).join(format!("{hex}.png"))
    }

    /// Decode a cached thumbnail, or `None` on any miss / corruption.
    pub fn load(&self, key: &CacheKey) -> Option<ColorImage> {
        let bytes = fs::read(self.entry_path(key)).ok()?;
        let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).ok()?;
        let rgba = img.to_rgba8();
        let size = [rgba.width() as usize, rgba.height() as usize];
        Some(ColorImage::from_rgba_unmultiplied(size, &rgba.into_raw()))
    }

    /// Write a thumbnail. `size` is `[w, h]`, `pixels` is RGBA8.
    ///
    /// Encodes to memory, writes a `.tmp` sibling, then renames: two GUI
    /// instances scanning the same dataset would otherwise race on the same
    /// path and leave a half-written PNG that every later run has to decode
    /// and reject.
    pub fn store(&self, key: &CacheKey, size: [usize; 2], pixels: &[u8]) {
        let path = self.entry_path(key);
        let Some(parent) = path.parent() else { return };
        if fs::create_dir_all(parent).is_err() {
            return;
        }
        let mut buf: Vec<u8> = Vec::new();
        if PngEncoder::new(&mut buf)
            .write_image(
                pixels,
                size[0] as u32,
                size[1] as u32,
                ExtendedColorType::Rgba8,
            )
            .is_err()
        {
            return;
        }
        let tmp = path.with_extension("png.tmp");
        if fs::write(&tmp, &buf).is_ok() && fs::rename(&tmp, &path).is_err() {
            let _ = fs::remove_file(&tmp);
        }
    }

    /// Total bytes currently held, for the settings readout.
    pub fn total_size(&self) -> u64 {
        self.entries().iter().map(|e| e.size).sum()
    }

    /// Remove every entry. Missing directory counts as success.
    pub fn clear(&self) -> std::io::Result<()> {
        match fs::remove_dir_all(&self.root) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }

    /// Enforce the configured budget: drop anything that has reached
    /// `max_age`, then, if still over `limit_bytes`, drop oldest-first until
    /// under it. `limit_bytes == 0` means "no size cap"; the caller maps a
    /// zero-day age setting to `None` the same way.
    ///
    /// "Oldest" is by write time, not access time — refreshing an mtime on
    /// every cache *hit* would turn a read-only fast path into a write per
    /// image, which is most of what the cache exists to avoid. The practical
    /// difference is that a dataset in daily use can still age out; it costs
    /// one slow open and re-populates.
    ///
    /// Runs off the UI thread (it stats every entry) and is best-effort
    /// throughout — a file another process holds open is simply skipped.
    pub fn prune(&self, limit_bytes: u64, max_age: Option<Duration>) {
        let mut entries = self.entries();
        let now = SystemTime::now();

        if let Some(max_age) = max_age {
            entries.retain(|e| {
                let expired = now
                    .duration_since(e.modified)
                    .map(|age| age >= max_age)
                    .unwrap_or(false);
                // Keep the entry in the list (so it still counts toward the
                // size budget) if the removal failed.
                !(expired && fs::remove_file(&e.path).is_ok())
            });
        }

        if limit_bytes == 0 {
            return;
        }
        let mut total: u64 = entries.iter().map(|e| e.size).sum();
        if total <= limit_bytes {
            return;
        }
        // Drop to 90% rather than exactly to the limit, so the next scan
        // doesn't immediately push us back over and prune again.
        let target = limit_bytes / 10 * 9;
        entries.sort_by_key(|e| e.modified);
        for e in &entries {
            if total <= target {
                break;
            }
            if fs::remove_file(&e.path).is_ok() {
                total = total.saturating_sub(e.size);
            }
        }
    }

    /// Flat listing of the shard directories. Hand-rolled two-level walk
    /// rather than pulling `walkdir` into the GUI for one function.
    fn entries(&self) -> Vec<CacheEntry> {
        let mut out = Vec::new();
        let Ok(shards) = fs::read_dir(&self.root) else {
            return out;
        };
        for shard in shards.flatten() {
            let Ok(files) = fs::read_dir(shard.path()) else {
                continue;
            };
            for f in files.flatten() {
                let Ok(md) = f.metadata() else { continue };
                if !md.is_file() {
                    continue;
                }
                out.push(CacheEntry {
                    path: f.path(),
                    size: md.len(),
                    modified: md.modified().unwrap_or(UNIX_EPOCH),
                });
            }
        }
        out
    }
}

struct CacheEntry {
    path: PathBuf,
    size: u64,
    modified: SystemTime,
}

/// Human-readable byte count for the settings readout.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit < UNITS.len() - 1 {
        v /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(mtime_ns: u128, size: u64) -> FileStamp {
        FileStamp { mtime_ns, size }
    }

    #[test]
    fn key_is_stable_for_same_inputs() {
        let p = Path::new("/data/set/a.png");
        assert_eq!(
            CacheKey::new(p, stamp(42, 100), 256),
            CacheKey::new(p, stamp(42, 100), 256)
        );
    }

    #[test]
    fn key_changes_with_every_component() {
        let p = Path::new("/data/set/a.png");
        let base = CacheKey::new(p, stamp(42, 100), 256);
        assert_ne!(
            base,
            CacheKey::new(Path::new("/data/set/b.png"), stamp(42, 100), 256)
        );
        assert_ne!(base, CacheKey::new(p, stamp(43, 100), 256));
        assert_ne!(base, CacheKey::new(p, stamp(42, 101), 256));
        assert_ne!(base, CacheKey::new(p, stamp(42, 100), 512));
    }

    #[test]
    fn key_hex_is_32_chars() {
        let hex = CacheKey::new(Path::new("/a"), stamp(1, 2), 256).hex();
        assert_eq!(hex.len(), 32);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn store_then_load_round_trips_pixels() {
        let dir =
            std::env::temp_dir().join(format!("fwaun-thumbcache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = ThumbCache { root: dir.clone() };
        let key = CacheKey::new(Path::new("/img.png"), stamp(7, 7), 256);

        assert!(cache.load(&key).is_none());
        // 2x1 RGBA: opaque red, half-transparent green.
        let pixels = [255u8, 0, 0, 255, 0, 255, 0, 128];
        cache.store(&key, [2, 1], &pixels);

        let img = cache.load(&key).expect("cached entry decodes");
        assert_eq!(img.size, [2, 1]);
        assert!(cache.total_size() > 0);

        cache.clear().unwrap();
        assert!(cache.load(&key).is_none());
        assert_eq!(cache.total_size(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_evicts_until_under_limit() {
        let dir =
            std::env::temp_dir().join(format!("fwaun-thumbcache-prune-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = ThumbCache { root: dir.clone() };
        // Ten 64x64 opaque entries; each PNG is well under a KB but the exact
        // size doesn't matter — prune only has to get the total under target.
        let pixels = vec![200u8; 64 * 64 * 4];
        for i in 0..10u32 {
            cache.store(
                &CacheKey::new(Path::new("/img.png"), stamp(i as u128, 1), 256),
                [64, 64],
                &pixels,
            );
        }
        let before = cache.total_size();
        assert!(before > 0);
        cache.prune(before / 2, None);
        assert!(cache.total_size() <= before / 2);

        // A zero limit means "no size cap", not "evict everything".
        let remaining = cache.total_size();
        cache.prune(0, None);
        assert_eq!(cache.total_size(), remaining);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn prune_drops_expired_entries() {
        let dir = std::env::temp_dir().join(format!("fwaun-thumbcache-age-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cache = ThumbCache { root: dir.clone() };
        let pixels = vec![0u8; 4];
        cache.store(
            &CacheKey::new(Path::new("/img.png"), stamp(1, 1), 256),
            [1, 1],
            &pixels,
        );
        assert!(cache.total_size() > 0);
        // Everything just written has reached a zero-length max age.
        cache.prune(0, Some(Duration::from_secs(0)));
        assert_eq!(cache.total_size(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn format_size_scales() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1536), "1.5 KB");
        assert_eq!(format_size(5 * 1024 * 1024), "5.0 MB");
    }
}
