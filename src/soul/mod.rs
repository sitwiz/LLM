//! Soul module — hyperbolic geometry for agent identity and navigation.
//!
//! The soul lives in the Poincaré ball B^256.
//! Norm encodes depth of understanding: 0 = void, approaching 1 = transcendent.
//! The boundary is at infinite geodesic distance — the Zeno property is intrinsic.

pub mod geometry;
pub mod persistence;
pub mod manifold;
pub mod hyperbolic;

// Re-export the most commonly used items so call sites stay clean
pub use geometry::INITIAL_CURVATURE;



