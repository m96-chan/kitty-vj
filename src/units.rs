//! The playable units, in one list.
//!
//! A cell effect and a pixel effect differ in the medium they draw into
//! and in nothing else that matters to an operator: both are Sources in
//! the taxonomy, both belong on a channel, both answer to a fader. The
//! graphics tier used to be a global mode, which meant the four faders
//! could mix cell effects with each other or pixel effects with each
//! other but never one against the other — a distinction the instrument
//! has no reason to enforce.
//!
//! So the medium is a property of the unit, not a state of the app. A
//! channel holds any unit; the compositor renders the pixel-medium ones
//! into the shared framebuffer and the cell-medium ones into cell
//! buffers, and the framebuffer lands under the cells at z=-1 as before.
//! There is no mode to be in.

/// Which surface a unit draws into. The only thing that separates the
/// two kinds, and the compositor's only reason to care.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Medium {
    /// Glyphs with colours — halfblock, ASCII, box drawing.
    Cells,
    /// Full RGB pixels over the kitty graphics protocol.
    Pixels,
}

/// A pixel-medium unit. Cell units are indices into the app's effect
/// vec, because they carry state (a plate rotation, a camera handle);
/// these are dispatched by kind for the same reason.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pix {
    Plasma,
    Tunnel,
    Stars,
    /// The live capture, drawn through the shared image sampler.
    Cam,
    /// A plate from the rotation, same sampler.
    Plate,
    Sparks,
    PxTunnel,
    Floor,
    Rings,
    MeshCube,
    MeshWire,
    MeshSpeaker,
}

impl Pix {
    /// Particle and mesh units composite over whatever is already in the
    /// framebuffer; the rest establish the picture. The compositor needs
    /// to know so a channel holding sparks does not erase the plate on
    /// the channel below it.
    pub fn additive(&self) -> bool {
        matches!(
            self,
            Pix::Sparks | Pix::PxTunnel | Pix::Floor | Pix::Rings | Pix::MeshWire
        )
    }

    /// How far a solid mesh ducks what is behind it, so the geometry
    /// carries the frame. 1.0 for everything that is not solid.
    pub fn dim_behind(&self) -> f64 {
        match self {
            Pix::MeshCube => crate::meshcube::PLATE_DIM,
            Pix::MeshSpeaker => crate::meshspeaker::PLATE_DIM,
            _ => 1.0,
        }
    }
}

/// One playable thing: a name for the HUD, and how to draw it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Unit {
    /// Index into the app's `effects` vec.
    Cell(usize),
    Pixel(Pix),
}

impl Unit {
    pub fn medium(&self) -> Medium {
        match self {
            Unit::Cell(_) => Medium::Cells,
            Unit::Pixel(_) => Medium::Pixels,
        }
    }
}

/// The pixel-medium units, in the order they appear after the cell ones.
/// Cell units are appended by the app, which knows how many effects it
/// built (a plate list may be empty).
pub const PIX_UNITS: [(&str, Pix); 12] = [
    ("PLASMA", Pix::Plasma),
    ("PXTUNNEL", Pix::Tunnel),
    ("STARS", Pix::Stars),
    ("PXCAM", Pix::Cam),
    ("PXPLATE", Pix::Plate),
    ("SPARKS3D", Pix::Sparks),
    ("TUBE", Pix::PxTunnel),
    ("FLOOR", Pix::Floor),
    ("RINGS", Pix::Rings),
    ("MCUBE", Pix::MeshCube),
    ("MWIRE", Pix::MeshWire),
    ("MSPKR", Pix::MeshSpeaker),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medium_follows_the_unit_not_a_mode() {
        assert_eq!(Unit::Cell(3).medium(), Medium::Cells);
        assert_eq!(Unit::Pixel(Pix::Plasma).medium(), Medium::Pixels);
    }

    #[test]
    fn additive_units_do_not_establish_the_picture() {
        // A channel holding sparks must not wipe the plate on the
        // channel below it; a channel holding plasma must.
        assert!(Pix::Sparks.additive());
        assert!(Pix::MeshWire.additive(), "the wire rig is a glow");
        assert!(!Pix::Plasma.additive());
        assert!(!Pix::MeshCube.additive(), "solid geometry replaces");
    }

    #[test]
    fn only_solid_meshes_duck_what_is_behind() {
        assert!(Pix::MeshCube.dim_behind() < 1.0);
        assert!(Pix::MeshSpeaker.dim_behind() < 1.0);
        assert_eq!(Pix::MeshWire.dim_behind(), 1.0);
        assert_eq!(Pix::Plasma.dim_behind(), 1.0);
    }

    #[test]
    fn pixel_unit_names_do_not_collide_with_the_cell_effects() {
        // Both media share one list and the scene director addresses it
        // by name, so a pixel unit called SPARKS would shadow the cell
        // effect of that name for every scene that asked for one.
        const CELL_EFFECT_NAMES: [&str; 9] = [
            "PULSE", "RAIN", "TUNNEL", "COLLAPSE", "CUBE", "SPARKS", "IMGDUST", "PLATE", "CAM",
        ];
        for (n, _) in PIX_UNITS {
            assert!(
                !CELL_EFFECT_NAMES.contains(&n),
                "{n} collides with a cell effect"
            );
        }
    }

    #[test]
    fn every_pixel_unit_is_named_once() {
        for (i, (n, _)) in PIX_UNITS.iter().enumerate() {
            assert!(!n.is_empty());
            assert!(
                PIX_UNITS.iter().skip(i + 1).all(|(m, _)| m != n),
                "duplicate name {n}"
            );
        }
    }
}
