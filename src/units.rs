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
//! There is no mode to be in. And since a combo bundles parts of either
//! medium, the compositor asks each expanded part which variant it is
//! rather than asking the slot for a single medium it may not have.

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
    /// The artwork on a tumbling framed panel — floats over the field.
    MeshPlate,
    /// The speaker rig as wireframe glow — layers instead of replacing.
    WireSpeaker,
    /// A video file, forward-decoded with a backspin ring (video.rs).
    Video,
}

impl Pix {
    /// Particle and mesh units composite over whatever is already in the
    /// framebuffer; the rest establish the picture. The compositor needs
    /// to know so a channel holding sparks does not erase the plate on
    /// the channel below it.
    /// `MeshPlate` is here not because it adds light but because it
    /// composites: it draws only its own silhouette, so the direct path
    /// leaves the field around the panel standing — routing it through
    /// the establish/blend path would fade the whole frame to black
    /// around it at full fader.
    pub fn additive(&self) -> bool {
        matches!(
            self,
            Pix::Sparks
                | Pix::PxTunnel
                | Pix::Floor
                | Pix::Rings
                | Pix::MeshWire
                | Pix::MeshPlate
                | Pix::WireSpeaker
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
    /// Index into the app's combo vec — a channel scene, expanding to
    /// several base units that share the channel's fader. Combos hold
    /// base units only; the compositor expands one level and no more.
    Combo(usize),
}

/// The pixel-medium units, in the order they appear after the cell ones.
/// Cell units are appended by the app, which knows how many effects it
/// built (a plate list may be empty).
pub const PIX_UNITS: [(&str, Pix); 15] = [
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
    ("MPLATE", Pix::MeshPlate),
    ("WSPKR", Pix::WireSpeaker),
    ("VIDEO", Pix::Video),
];

#[cfg(test)]
mod tests {
    use super::*;

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
        for (n, _) in PIX_UNITS {
            assert!(
                !crate::effects::CELL_NAMES.contains(&n),
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
