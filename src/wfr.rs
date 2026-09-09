//! WFR1 — the baked woofer, decoded.
//!
//! The accurate mesh was never locked in the missing FBX: EasyPngVJ
//! ships it decoded and quantised in `assets/models/woofer.js` ("woofer
//! print 04", 8,774 vertices / 11,944 triangles), because OBS opens the
//! page from disk where `fetch()` is blocked. The base64 there wraps
//! this binary layout, written by its `tools/build_model.py`:
//!
//! ```text
//! 0x00  "WFR1"
//! 0x04  u32 LE  vertex count
//! 0x08  u32 LE  index count
//! 0x0c  f32 LE  scale
//! 0x10  i16 LE × 3n   positions, /32767 · scale
//!       i8      × 3n   normals, /127        (skipped: wire wants edges)
//!       (pad to 4)
//!       u16 LE  × ni   triangle indices
//! ```
//!
//! The copy here (`assets/models/woofer.wfr`, embedded at compile time)
//! is that binary with the base64 peeled off — byte-identical payload.
//!
//! ## Why it is decimated before drawing
//!
//! Twelve thousand triangles were free on the GPU the original ran on;
//! on a software rasteriser they are ~6 ms *per instance*, and the wire
//! rig draws three. So the mesh is thinned by deterministic vertex
//! clustering: vertices snap to a grid over the bounding box, clusters
//! average, and a triangle survives if its corners land in three
//! distinct cells. Crude next to a real edge-collapse, but stable,
//! seed-free, and honest about what it is — and a wireframe this dense
//! reads as shimmer either way; the silhouette is what the print model
//! actually contributes.
//!
//! Determinism note: clustering iterates a `BTreeMap`, not a `HashMap`
//! — hash iteration order changes per process, which would reorder
//! vertices and change frame bytes between two runs of the same set.

use crate::raster::Vec3;

/// The decoded mesh: positions and indexed triangles. Normals are in
/// the payload but not kept — the wire pass shades edges, not surfaces.
pub struct Baked {
    pub verts: Vec<Vec3>,
    pub tris: Vec<[u32; 3]>,
}

fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Parse a WFR1 payload. `None` on anything malformed — the caller
/// falls back to the generated woofer rather than crashing a set.
pub fn decode(b: &[u8]) -> Option<Baked> {
    if b.len() < 16 || &b[..4] != b"WFR1" {
        return None;
    }
    let nv = u32le(b, 4) as usize;
    let ni = u32le(b, 8) as usize;
    let scale = f32::from_le_bytes([b[12], b[13], b[14], b[15]]) as f64;
    if !scale.is_finite() || nv == 0 || ni == 0 || !ni.is_multiple_of(3) {
        return None;
    }
    let pos_at = 16usize;
    let nrm_at = pos_at.checked_add(nv.checked_mul(6)?)?;
    let idx_at = (nrm_at.checked_add(nv.checked_mul(3)?)? + 3) & !3;
    let end = idx_at.checked_add(ni.checked_mul(2)?)?;
    if b.len() < end {
        return None;
    }
    let mut verts = Vec::with_capacity(nv);
    for i in 0..nv {
        let c = |k: usize| {
            let o = pos_at + (i * 3 + k) * 2;
            i16::from_le_bytes([b[o], b[o + 1]]) as f64 / 32767.0 * scale
        };
        verts.push(Vec3 {
            x: c(0),
            y: c(1),
            z: c(2),
        });
    }
    let mut tris = Vec::with_capacity(ni / 3);
    for t in 0..ni / 3 {
        let ix = |k: usize| {
            let o = idx_at + (t * 3 + k) * 2;
            u16::from_le_bytes([b[o], b[o + 1]]) as u32
        };
        let (a, bb, c) = (ix(0), ix(1), ix(2));
        if a as usize >= nv || bb as usize >= nv || c as usize >= nv {
            return None;
        }
        tris.push([a, bb, c]);
    }
    Some(Baked { verts, tris })
}

/// Thin by vertex clustering on a `g³` grid over the bounding box.
/// Deterministic: cluster order follows grid coordinates, triangle
/// order follows the source mesh.
pub fn cluster(src: &Baked, g: u32) -> Baked {
    let g = g.max(2);
    let mut lo = Vec3 {
        x: f64::MAX,
        y: f64::MAX,
        z: f64::MAX,
    };
    let mut hi = Vec3 {
        x: f64::MIN,
        y: f64::MIN,
        z: f64::MIN,
    };
    for v in &src.verts {
        lo.x = lo.x.min(v.x);
        lo.y = lo.y.min(v.y);
        lo.z = lo.z.min(v.z);
        hi.x = hi.x.max(v.x);
        hi.y = hi.y.max(v.y);
        hi.z = hi.z.max(v.z);
    }
    let span = |a: f64, b: f64| if b - a > 1e-12 { b - a } else { 1.0 };
    let (sx, sy, sz) = (span(lo.x, hi.x), span(lo.y, hi.y), span(lo.z, hi.z));
    let cell_of = |v: &Vec3| {
        let q = |p: f64, l: f64, s: f64| (((p - l) / s * g as f64) as u32).min(g - 1);
        (q(v.x, lo.x, sx), q(v.y, lo.y, sy), q(v.z, lo.z, sz))
    };

    let mut cells: std::collections::BTreeMap<(u32, u32, u32), (Vec3, u32)> =
        std::collections::BTreeMap::new();
    for v in &src.verts {
        let e = cells.entry(cell_of(v)).or_insert((Vec3::ZERO, 0));
        e.0 = e.0 + *v;
        e.1 += 1;
    }
    let mut id = std::collections::BTreeMap::new();
    let mut verts = Vec::with_capacity(cells.len());
    for (key, (sum, n)) in &cells {
        id.insert(*key, verts.len() as u32);
        verts.push(*sum * (1.0 / *n as f64));
    }
    let mut seen = std::collections::HashSet::new();
    let mut tris = Vec::new();
    for &[a, b, c] in &src.tris {
        let ka = cell_of(&src.verts[a as usize]);
        let kb = cell_of(&src.verts[b as usize]);
        let kc = cell_of(&src.verts[c as usize]);
        if ka == kb || kb == kc || ka == kc {
            continue; // collapsed to an edge or a point
        }
        let t = [id[&ka], id[&kb], id[&kc]];
        if seen.insert(t) {
            tris.push(t);
        }
    }
    Baked { verts, tris }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAYLOAD: &[u8] = include_bytes!("../assets/models/woofer.wfr");

    #[test]
    fn the_shipped_payload_decodes_to_the_documented_counts() {
        let b = decode(PAYLOAD).expect("woofer.wfr must decode");
        assert_eq!(b.verts.len(), 8774);
        assert_eq!(b.tris.len(), 11944);
        // Model-space sanity: unit-ish radius, mouth at z = 0, body
        // down -z — the wire mode's cone test depends on this frame.
        for v in &b.verts {
            assert!(v.x.hypot(v.y) < 1.05);
            assert!((-0.9..=0.0).contains(&v.z), "z out of band: {}", v.z);
        }
    }

    #[test]
    fn malformed_payloads_are_refused_not_parsed() {
        assert!(decode(&[]).is_none());
        assert!(decode(b"WFR1").is_none(), "header alone");
        let mut bad = PAYLOAD.to_vec();
        bad[0] = b'X';
        assert!(decode(&bad).is_none(), "magic");
        assert!(decode(&PAYLOAD[..PAYLOAD.len() - 8]).is_none(), "truncated");
    }

    #[test]
    fn clustering_is_deterministic_and_actually_thins() {
        let b = decode(PAYLOAD).unwrap();
        let a1 = cluster(&b, 18);
        let a2 = cluster(&b, 18);
        assert_eq!(a1.verts.len(), a2.verts.len());
        assert_eq!(a1.tris, a2.tris);
        assert!(
            a1.verts
                .iter()
                .zip(&a2.verts)
                .all(|(p, q)| p.x == q.x && p.y == q.y && p.z == q.z),
            "vertex order or averaging changed between runs"
        );
        assert!(
            a1.tris.len() < b.tris.len() / 2,
            "grid 18 barely thinned: {}",
            a1.tris.len()
        );
        let coarse = cluster(&b, 12);
        assert!(coarse.tris.len() < a1.tris.len());
        // Every index in range.
        for t in &a1.tris {
            assert!(t.iter().all(|&i| (i as usize) < a1.verts.len()));
        }
    }
}
