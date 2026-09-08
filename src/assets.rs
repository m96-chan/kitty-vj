//! Asset loading. Same layout as EasyPngVJ's `tools/build_assets.sh`
//! output: `assets/bg` (16:9 plates), `assets/poster` (2:3). Images are
//! decoded once at startup and pre-downscaled; per-frame sampling happens
//! in the plate effect.

use std::path::{Path, PathBuf};

use image::RgbImage;
use image::imageops::FilterType;

/// Longest edge after pre-downscale. Halfblock output tops out around
/// 400x300 "pixels" on a fullscreen terminal, so this keeps sampling
/// cache-friendly without visible loss.
const MAX_EDGE: u32 = 640;

pub struct Plate {
    pub name: String,
    pub img: RgbImage,
}

/// Load every PNG/JPEG under `root`, `root/bg`, and `root/poster`.
/// Sorted by filename so the rotation order is deterministic.
pub fn load(root: &Path) -> Vec<Plate> {
    let mut files: Vec<PathBuf> = Vec::new();
    for dir in [root.to_path_buf(), root.join("bg"), root.join("poster")] {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let p = e.path();
                let ext = p
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase());
                if matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
                    files.push(p);
                }
            }
        }
    }
    files.sort();
    files.dedup();

    files
        .into_iter()
        .filter_map(|p| {
            let img = image::open(&p).ok()?.into_rgb8();
            let (w, h) = img.dimensions();
            let img = if w.max(h) > MAX_EDGE {
                let s = MAX_EDGE as f64 / w.max(h) as f64;
                image::imageops::resize(
                    &img,
                    (w as f64 * s) as u32,
                    (h as f64 * s) as u32,
                    FilterType::Triangle,
                )
            } else {
                img
            };
            let name = p
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            Some(Plate { name, img })
        })
        .collect()
}
