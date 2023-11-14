use fast_surface_nets::glam::Vec3A;
use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use ilattice::prelude::*;
use rand::{Rng, rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::io::Result;
use std::time;

// The un-padded chunk side, it will become 18*18*18 with padding
const UNPADDED_CHUNK_SIDE: u32 = 16_u32;
const DEFAULT_SDF_VALUE: f32 = 999.0;

type PaddedChunkShape = ConstShape3u32<
    { UNPADDED_CHUNK_SIDE + 2 },
    { UNPADDED_CHUNK_SIDE + 2 },
    { UNPADDED_CHUNK_SIDE + 2 },
>;

type Extent3i = Extent<IVec3>;

/// Convert the content of the buffers to obj_exporter format, write the result to file
fn write_mesh_to_obj_file(name: String, buffers: &[(Vec3A, SurfaceNetsBuffer)]) -> Result<()> {
    let filename = format!("{}.obj", name);

    let (mut vertices, mut normals, mut shapes) = {
        let sum = buffers
            .iter()
            .fold((0usize, 0usize, 0usize), |(p, n, i), (_, buffer)| {
                (
                    p + buffer.positions.len(),
                    n + buffer.normals.len(),
                    i + buffer.indices.len(),
                )
            });
        (
            Vec::<obj_exporter::Vertex>::with_capacity(sum.0),
            Vec::<obj_exporter::Vertex>::with_capacity(sum.1),
            Vec::<obj_exporter::Shape>::with_capacity(sum.2 / 3),
        )
    };
    let mut vertex_offset = 0_usize;
    let mut normal_offset = 0_usize;

    for (buffer_offset, buffer) in buffers.iter() {
        vertices.append(
            &mut buffer
                .positions
                .iter()
                .map(|&[x, y, z]| obj_exporter::Vertex {
                    x: x as f64 + buffer_offset.x as f64,
                    y: y as f64 + buffer_offset.y as f64,
                    z: z as f64 + buffer_offset.z as f64,
                })
                .collect::<Vec<_>>(),
        );
        normals.append(
            &mut buffer
                .normals
                .iter()
                .map(|&[x, y, z]| obj_exporter::Vertex {
                    x: x as f64,
                    y: y as f64,
                    z: z as f64,
                })
                .collect::<Vec<_>>(),
        );
        shapes.append(
            &mut buffer
                .indices
                .chunks(3)
                .map(|tri| obj_exporter::Shape {
                    primitive: obj_exporter::Primitive::Triangle(
                        (
                            vertex_offset + tri[0] as usize,
                            None,
                            Some(normal_offset + tri[0] as usize),
                        ),
                        (
                            vertex_offset + tri[1] as usize,
                            None,
                            Some(normal_offset + tri[1] as usize),
                        ),
                        (
                            vertex_offset + tri[2] as usize,
                            None,
                            Some(normal_offset + tri[2] as usize),
                        ),
                    ),
                    groups: vec![],
                    smoothing_groups: vec![],
                })
                .collect::<Vec<_>>(),
        );
        vertex_offset = vertices.len();
        normal_offset = normals.len();
    }
    let number_of_tris = shapes.len();

    obj_exporter::export_to_file(
        &obj_exporter::ObjSet {
            material_library: None,
            objects: vec![obj_exporter::Object {
                name,
                vertices,
                normals,
                geometry: vec![obj_exporter::Geometry {
                    material_name: None,
                    shapes,
                }],
                tex_vertices: vec![],
            }],
        },
        filename.clone(),
    )?;
    println!(
        "Succcessfully wrote {} triangles to {}",
        number_of_tris, &filename
    );
    Ok(())
}

// There is no need to waste CPU cycles calculating the sdf values of a tiny shape located on the other side of the map. This is
// an attempt to group sdf functions by their footprints.
//
// For example, if the sdf describes a sphere with radius 9 centered at [0,0,0] the function will not affect anything outside the
// AABB [-10,-10,-10] <-> [10,10,10] and can thus be ignored at coordinates outside that AABB.
//
// Note that this only works if you are interested in the transition point between negative and positive (inside vs outside) of
// a shape. This might be sub-optimal if using advanced blending/combination functions that uses data "far away" from the shapes.
struct Sphere {
    /// The extent of this sdf, the sdf is completely inside this AABB
    extent: Extent3i,
    origin: Vec3A,
    radius: f32,
}

/// Generate the data of a single chunk by merging the sdf value (of the shapes intersecting the chunk) with a simple .min()
/// function.
fn generate_and_process_chunk(
    padded_chunk_extent: Extent3i,
    spheres: &[Sphere],
) -> Option<(Vec3A, SurfaceNetsBuffer)> {
    // filter out every sdf that does not affect this chunk
    let intersecting_spheres: Vec<_> = spheres
        .iter()
        .filter(|sphere| !sphere.extent.intersection(&padded_chunk_extent).is_empty())
        .collect();

    // No need to proceed if there is no data to process
    if intersecting_spheres.is_empty() {
        return None;
    }

    let mut array = { [DEFAULT_SDF_VALUE; PaddedChunkShape::SIZE as usize] };

    let mut some_neg_or_zero_found = false;
    let mut some_pos_found = false;

    padded_chunk_extent.iter3().for_each(|pwo| {
        // pwo (PointWithOffset) is the voxel coordinate with the offset added, (i.e world coordinate system).

        // Point With Offset as float
        let pwof = pwo.as_vec3a();

        // p is the voxel coordinate without the padded chunk offset (i.e. padded chunk coordinate system)
        //
        // Note that the chunk is padded(1) so the p coordinate goes from [0;3] to [UNPADDED_CHUNK_SIDE+2;3].
        //
        // p=[1,1,1] (chunk coordinate system) correlates to unpadded_chunk_extent.minimum (world coordinate system)
        let p = pwo - padded_chunk_extent.minimum;

        // *v is the value of a voxel at p
        let v =
            &mut array[PaddedChunkShape::linearize([p.x as u32, p.y as u32, p.z as u32]) as usize];

        for sphere in intersecting_spheres.iter() {
            // You could use an additional test here to see if the individual voxel itself is contained within the shape extent.
            // This might result in a speed gain, specially for complex sdf functions.
            // if !sphere.extent.contains(pwo) {continue}

            // Use a simple .min() function to merge sdf values
            *v = v.min((sphere.origin - pwof).length() - sphere.radius);
        }

        if *v <= 0.0 {
            some_neg_or_zero_found = true;
        } else {
            some_pos_found = true;
        }
    });
    if some_pos_found && some_neg_or_zero_found {
        // A combination of positive and negative values found - mesh this chunk
        let mut sn_buffer = SurfaceNetsBuffer::default();

        surface_nets(
            &array,
            &PaddedChunkShape {},
            [0; 3],
            [UNPADDED_CHUNK_SIDE + 1; 3],
            &mut sn_buffer,
        );

        if sn_buffer.positions.is_empty() {
            // No vertices were generated by this chunk, ignore it
            None
        } else {
            Some((padded_chunk_extent.minimum.as_vec3a(), sn_buffer))
        }
    } else {
        // Only positive or only negative values found, so no mesh will be generated from this chunk - ignore it
        None
    }
}

/// Generate some example data and then spawn off rayon thread tasks, one for each chunk
fn generate_chunks() -> Vec<(Vec3A, SurfaceNetsBuffer)> {
    // Create a chunk extent cube with a side of 10 chunks, and each chunk side is 16 voxels.
    //
    // Note that the size of this extent is measured in chunks, not voxels
    let chunks_extent = Extent3i::from_min_and_lub(IVec3::from([-5; 3]), IVec3::from([5; 3]));

    let mut rng: StdRng = SeedableRng::from_seed([42; 32]);

    let spheres: Vec<Sphere> = (0..501)
        .map(|_| {
            let origin = Vec3A::new(
                rng.gen_range(-74.0..74.0),
                rng.gen_range(-74.0..74.0),
                rng.gen_range(-74.0..74.0),
            );
            let radius = rng.gen_range(2.5..5.0);
            let extent = Extent::from_min_and_lub(origin, origin)
                .padded(radius)
                .containing_integer_extent();
            Sphere {
                extent,
                origin,
                radius,
            }
        })
        .collect();

    let now = time::Instant::now();

    let unpadded_chunk_shape = IVec3::from([UNPADDED_CHUNK_SIDE as i32; 3]);
    let chunks: Vec<_> = chunks_extent
        .iter3()
        .par_bridge()
        .filter_map(|p| {
            let chunk_min = p * unpadded_chunk_shape;

            generate_and_process_chunk(
                Extent3i::from_min_and_shape(chunk_min, unpadded_chunk_shape).padded(1),
                &spheres,
            )
        })
        .collect();
    println!(
        "Generated and meshed {} chunks in {:?}",
        chunks.len(),
        now.elapsed()
    );
    chunks
}

fn main() -> Result<()> {
    let chunk_buffers = generate_chunks();
    write_mesh_to_obj_file("mesh".to_string(), &chunk_buffers)?;
    Ok(())
}
