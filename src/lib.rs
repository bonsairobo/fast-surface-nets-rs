//! A fast, chunk-friendly implementation of Naive Surface Nets on regular grids.
//!
//! Surface Nets is an algorithm for extracting an isosurface mesh from a [signed distance
//! field](https://en.wikipedia.org/wiki/Signed_distance_function) sampled on a regular grid. It is nearly the same as Dual
//! Contouring, but instead of using hermite (derivative) data to estimate surface points, Surface Nets will do a simpler form
//! of interpolation (average) between points where the isosurface crosses voxel cube edges.
//!
//! Benchmarks show that [`surface_nets`] generates about 20 million triangles per second on a single core of a 2.5 GHz Intel
//! Core i7. This implementation achieves high performance by using small lookup tables and SIMD acceleration provided by `glam`
//! when doing 3D floating point vector math. To run the benchmarks yourself, `cd bench/ && cargo bench`.
//!
//! High-quality surface normals are estimated by:
//!
//! 1. calculating SDF derivatives using central differencing
//! 2. using bilinear interpolation of SDF derivatives along voxel cube edges
//!
//! When working with sparse data sets, [`surface_nets`] can generate meshes for array chunks that fit together seamlessly. This
//! works because faces are not generated on the positive boundaries of a chunk. One must only apply a translation of the mesh
//! into proper world coordinates for the given chunk.
//!
//! # Example Code
//!
//! ```
//! use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
//! use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
//!
//! // A 16^3 chunk with 1-voxel boundary padding.
//! type ChunkShape = ConstShape3u32<18, 18, 18>;
//!
//! // This chunk will cover just a single octant of a sphere SDF (radius 15).
//! let mut sdf = [1.0; ChunkShape::SIZE as usize];
//! for i in 0u32..(ChunkShape::SIZE) {
//!     let [x, y, z] = ChunkShape::delinearize(i);
//!     sdf[i as usize] = ((x * x + y * y + z * z) as f32).sqrt() - 15.0;
//! }
//!
//! let mut buffer = SurfaceNetsBuffer::default();
//! surface_nets(&sdf, &ChunkShape {}, [0; 3], [17; 3], &mut buffer);
//!
//! // Some triangles were generated.
//! assert!(!buffer.indices.is_empty());
//! ```

pub use ndshape;

use glam::{const_vec3a, Vec3A, Vec3Swizzles};
use ndshape::Shape;

pub trait SignedDistance: Into<f32> + Copy {
    fn is_negative(self) -> bool;
}

impl SignedDistance for f32 {
    fn is_negative(self) -> bool {
        self < 0.0
    }
}

/// The output buffers used by `surface_nets`. These buffers can be reused to avoid reallocating memory.
#[derive(Default)]
pub struct SurfaceNetsBuffer {
    /// The triangle mesh positions.
    ///
    /// These are in array-local coordinates, i.e. at array position `(x, y, z)`, the vertex position would be `(x, y, z) +
    /// centroid` if the isosurface intersects that voxel.
    pub positions: Vec<[f32; 3]>,
    /// The triangle mesh normals.
    ///
    /// The normals are **not** normalized, since that is done most efficiently on the GPU.
    pub normals: Vec<[f32; 3]>,
    /// The triangle mesh indices.
    pub indices: Vec<u32>,

    /// Local 3D array coordinates of every voxel that intersects the isosurface.
    pub surface_points: Vec<[u32; 3]>,
    /// Stride of every voxel that intersects the isosurface. Can be used for efficient post-processing.
    pub surface_strides: Vec<u32>,
    /// Used to map back from voxel stride to vertex index.
    pub stride_to_index: Vec<u32>,
}

impl SurfaceNetsBuffer {
    /// Clears all of the buffers, but keeps the memory allocated for reuse.
    fn reset(&mut self, array_size: usize) {
        self.positions.clear();
        self.normals.clear();
        self.indices.clear();
        self.surface_points.clear();
        self.surface_strides.clear();

        // Just make sure this buffer is big enough, whether or not we've used it before.
        self.stride_to_index.resize(array_size, 0);
    }
}

/// The Naive Surface Nets smooth voxel meshing algorithm.
///
/// Extracts an isosurface mesh from the [signed distance field](https://en.wikipedia.org/wiki/Signed_distance_function) `sdf`.
/// Each value in the field determines how close that point is to the isosurface. Negative values are considered "interior" of
/// the surface volume, and positive values are considered "exterior." These lattice points will be considered corners of unit
/// cubes. For each unit cube, at most one isosurface vertex will be estimated, as below, where `p` is a positive corner value,
/// `n` is a negative corner value, `s` is an isosurface vertex, and `|` or `-` are mesh polygons connecting the vertices.
///
/// ```text
/// p   p   p   p
///   s---s
/// p | n | p   p
///   s   s---s
/// p | n   n | p
///   s---s---s
/// p   p   p   p
/// ```
///
/// The set of corners sampled is exactly the set of points in `[min, max]`. `sdf` must contain all of those points.
///
/// Note that the scheme illustrated above implies that chunks must be padded with a 1-voxel border copied from neighboring
/// voxels in order to connect seamlessly.
pub fn surface_nets<T, S>(
    sdf: &[T],
    shape: &S,
    min: [u32; 3],
    max: [u32; 3],
    output: &mut SurfaceNetsBuffer,
) where
    T: SignedDistance,
    S: Shape<u32, 3>,
{
    // SAFETY
    // Make sure the slice matches the shape before we start using get_unchecked.
    assert!(shape.linearize(min) <= shape.linearize(max));
    assert!((shape.linearize(max) as usize) < sdf.len());

    output.reset(sdf.len());

    estimate_surface(sdf, shape, min, max, output);
    make_all_quads(sdf, shape, min, max, output);
}

// Find all vertex positions and normals. Also generate a map from grid position to vertex index to be used to look up vertices
// when generating quads.
fn estimate_surface<T, S>(
    sdf: &[T],
    shape: &S,
    [minx, miny, minz]: [u32; 3],
    [maxx, maxy, maxz]: [u32; 3],
    output: &mut SurfaceNetsBuffer,
) where
    T: SignedDistance,
    S: Shape<u32, 3>,
{
    for z in minz..maxz {
        for y in miny..maxy {
            for x in minx..maxx {
                let stride = shape.linearize([x, y, z]);
                let p = Vec3A::new(x as f32, y as f32, z as f32);
                if estimate_surface_in_cube(sdf, shape, p, stride, output) {
                    output.stride_to_index[stride as usize] = output.positions.len() as u32 - 1;
                    output.surface_points.push([x, y, z]);
                    output.surface_strides.push(stride);
                }
            }
        }
    }
}

// Consider the grid-aligned cube where `point` is the minimal corner. Find a point inside this cube that is approximately on
// the isosurface.
//
// This is done by estimating, for each cube edge, where the isosurface crosses the edge (if it does at all). Then the estimated
// surface point is the average of these edge crossings.
fn estimate_surface_in_cube<T, S>(
    sdf: &[T],
    shape: &S,
    p: Vec3A,
    min_corner_stride: u32,
    output: &mut SurfaceNetsBuffer,
) -> bool
where
    T: SignedDistance,
    S: Shape<u32, 3>,
{
    // Get the signed distance values at each corner of this cube.
    let mut corner_dists = [0f32; 8];
    let mut num_negative = 0;
    for (i, dist) in corner_dists.iter_mut().enumerate() {
        let corner_stride = min_corner_stride + shape.linearize(CUBE_CORNERS[i]);
        let d = *unsafe { sdf.get_unchecked(corner_stride as usize) };
        *dist = d.into();
        if d.is_negative() {
            num_negative += 1;
        }
    }

    if num_negative == 0 || num_negative == 8 {
        // No crossings.
        return false;
    }

    let c = centroid_of_edge_intersections(&corner_dists);

    output.positions.push((p + c).into());
    output.normals.push(sdf_gradient(&corner_dists, c).into());

    true
}

fn centroid_of_edge_intersections(dists: &[f32; 8]) -> Vec3A {
    let mut count = 0;
    let mut sum = Vec3A::ZERO;
    for &[corner1, corner2] in CUBE_EDGES.iter() {
        let d1 = dists[corner1 as usize];
        let d2 = dists[corner2 as usize];
        if (d1 < 0.0) != (d2 < 0.0) {
            count += 1;
            sum += estimate_surface_edge_intersection(corner1, corner2, d1, d2);
        }
    }

    sum / count as f32
}

// Given two cube corners, find the point between them where the SDF is zero. (This might not exist).
fn estimate_surface_edge_intersection(
    corner1: u32,
    corner2: u32,
    value1: f32,
    value2: f32,
) -> Vec3A {
    let interp1 = value1 / (value1 - value2);
    let interp2 = 1.0 - interp1;

    interp2 * CUBE_CORNER_VECTORS[corner1 as usize]
        + interp1 * CUBE_CORNER_VECTORS[corner2 as usize]
}

/// Calculate the normal as the gradient of the distance field. Don't bother making it a unit vector, since we'll do that on the
/// GPU.
///
/// For each dimension, there are 4 cube edges along that axis. This will do bilinear interpolation between the differences
/// along those edges based on the position of the surface (s).
fn sdf_gradient(dists: &[f32; 8], s: Vec3A) -> Vec3A {
    let p00 = Vec3A::new(dists[0b001], dists[0b010], dists[0b100]);
    let n00 = Vec3A::new(dists[0b000], dists[0b000], dists[0b000]);

    let p10 = Vec3A::new(dists[0b101], dists[0b011], dists[0b110]);
    let n10 = Vec3A::new(dists[0b100], dists[0b001], dists[0b010]);

    let p01 = Vec3A::new(dists[0b011], dists[0b110], dists[0b101]);
    let n01 = Vec3A::new(dists[0b010], dists[0b100], dists[0b001]);

    let p11 = Vec3A::new(dists[0b111], dists[0b111], dists[0b111]);
    let n11 = Vec3A::new(dists[0b110], dists[0b101], dists[0b011]);

    // Each dimension encodes an edge delta, giving 12 in total.
    let d00 = p00 - n00; // Edges (0b00x, 0b0y0, 0bz00)
    let d10 = p10 - n10; // Edges (0b10x, 0b0y1, 0bz10)
    let d01 = p01 - n01; // Edges (0b01x, 0b1y0, 0bz01)
    let d11 = p11 - n11; // Edges (0b11x, 0b1y1, 0bz11)

    let neg = const_vec3a!([1.0; 3]) - s;

    // Do bilinear interpolation between 4 edges in each dimension.
    neg.yzx() * neg.zxy() * d00
        + neg.yzx() * s.zxy() * d10
        + s.yzx() * neg.zxy() * d01
        + s.yzx() * s.zxy() * d11
}

// For every edge that crosses the isosurface, make a quad between the "centers" of the four cubes touching that surface. The
// "centers" are actually the vertex sitions found earlier. Also, make sure the triangles are facing the right way. See the
// comments on `maybe_make_quad` to help with understanding the indexing.
fn make_all_quads<T, S>(
    sdf: &[T],
    shape: &S,
    [minx, miny, minz]: [u32; 3],
    [maxx, maxy, maxz]: [u32; 3],
    output: &mut SurfaceNetsBuffer,
) where
    T: SignedDistance,
    S: Shape<u32, 3>,
{
    let xyz_strides = [
        shape.linearize([1, 0, 0]) as usize,
        shape.linearize([0, 1, 0]) as usize,
        shape.linearize([0, 0, 1]) as usize,
    ];

    for (&[x, y, z], &p_stride) in output
        .surface_points
        .iter()
        .zip(output.surface_strides.iter())
    {
        let p_stride = p_stride as usize;

        // Do edges parallel with the X axis
        if y != miny && z != minz && x != maxx - 1 {
            maybe_make_quad(
                sdf,
                &output.stride_to_index,
                &output.positions,
                p_stride,
                p_stride + xyz_strides[0],
                xyz_strides[1],
                xyz_strides[2],
                &mut output.indices,
            );
        }
        // Do edges parallel with the Y axis
        if x != minx && z != minz && y != maxy - 1 {
            maybe_make_quad(
                sdf,
                &output.stride_to_index,
                &output.positions,
                p_stride,
                p_stride + xyz_strides[1],
                xyz_strides[2],
                xyz_strides[0],
                &mut output.indices,
            );
        }
        // Do edges parallel with the Z axis
        if x != minx && y != miny && z != maxz - 1 {
            maybe_make_quad(
                sdf,
                &output.stride_to_index,
                &output.positions,
                p_stride,
                p_stride + xyz_strides[2],
                xyz_strides[0],
                xyz_strides[1],
                &mut output.indices,
            );
        }
    }
}

// Construct a quad in the dual graph of the SDF lattice.
//
// The surface point s was found somewhere inside of the cube with minimal corner p1.
//
//       x ---- x
//      /      /|
//     x ---- x |
//     |   s  | x
//     |      |/
//    p1 --- p2
//
// And now we want to find the quad between p1 and p2 where s is a corner of the quad.
//
//          s
//         /|
//        / |
//       |  |
//   p1  |  |  p2
//       | /
//       |/
//
// If A is (of the three grid axes) the axis between p1 and p2,
//
//       A
//   p1 ---> p2
//
// then we must find the other 3 quad corners by moving along the other two axes (those orthogonal to A) in the negative
// directions; these are axis B and axis C.
#[allow(clippy::too_many_arguments)]
fn maybe_make_quad<T>(
    sdf: &[T],
    stride_to_index: &[u32],
    positions: &[[f32; 3]],
    p1: usize,
    p2: usize,
    axis_b_stride: usize,
    axis_c_stride: usize,
    indices: &mut Vec<u32>,
) where
    T: SignedDistance,
{
    let d1 = unsafe { sdf.get_unchecked(p1) };
    let d2 = unsafe { sdf.get_unchecked(p2) };
    let negative_face = match (d1.is_negative(), d2.is_negative()) {
        (true, false) => false,
        (false, true) => true,
        _ => return, // No face.
    };

    // The triangle points, viewed face-front, look like this:
    // v1 v3
    // v2 v4
    let v1 = stride_to_index[p1];
    let v2 = stride_to_index[p1 - axis_b_stride];
    let v3 = stride_to_index[p1 - axis_c_stride];
    let v4 = stride_to_index[p1 - axis_b_stride - axis_c_stride];
    let (pos1, pos2, pos3, pos4) = (
        Vec3A::from(positions[v1 as usize]),
        Vec3A::from(positions[v2 as usize]),
        Vec3A::from(positions[v3 as usize]),
        Vec3A::from(positions[v4 as usize]),
    );
    // Split the quad along the shorter axis, rather than the longer one.
    let quad = if pos1.distance_squared(pos4) < pos2.distance_squared(pos3) {
        if negative_face {
            [v1, v4, v2, v1, v3, v4]
        } else {
            [v1, v2, v4, v1, v4, v3]
        }
    } else if negative_face {
        [v2, v3, v4, v2, v1, v3]
    } else {
        [v2, v4, v3, v2, v3, v1]
    };
    indices.extend_from_slice(&quad);
}

const CUBE_CORNERS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];
const CUBE_CORNER_VECTORS: [Vec3A; 8] = [
    const_vec3a!([0.0, 0.0, 0.0]),
    const_vec3a!([1.0, 0.0, 0.0]),
    const_vec3a!([0.0, 1.0, 0.0]),
    const_vec3a!([1.0, 1.0, 0.0]),
    const_vec3a!([0.0, 0.0, 1.0]),
    const_vec3a!([1.0, 0.0, 1.0]),
    const_vec3a!([0.0, 1.0, 1.0]),
    const_vec3a!([1.0, 1.0, 1.0]),
];
const CUBE_EDGES: [[u32; 2]; 12] = [
    [0b000, 0b001],
    [0b000, 0b010],
    [0b000, 0b100],
    [0b001, 0b011],
    [0b001, 0b101],
    [0b010, 0b011],
    [0b010, 0b110],
    [0b011, 0b111],
    [0b100, 0b101],
    [0b100, 0b110],
    [0b101, 0b111],
    [0b110, 0b111],
];
