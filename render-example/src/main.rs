use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
};
use obj_exporter::{export_to_file, Geometry, ObjSet, Object, Primitive, Shape, Vertex};

fn main() {
    App::build()
        .add_plugins(DefaultPlugins)
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let mut samples = [1.0; SampleShape::SIZE as usize];
    for i in 0u32..(SampleShape::SIZE) {
        let p = into_domain(64, SampleShape::delinearize(i));
        samples[i as usize] = sdf(p);
    }

    // Do a single run first to allocate the buffer to the right size.
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&samples, &SampleShape {}, [0; 3], [65; 3], &mut buffer);

    let num_vertices = buffer.positions.len();

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float3(buffer.positions.clone()),
    );
    render_mesh.set_attribute(
        "Vertex_Normal",
        VertexAttributeValues::Float3(buffer.normals.clone()),
    );
    render_mesh.set_attribute(
        "Vertex_Uv",
        VertexAttributeValues::Float2(vec![[0.0; 2]; num_vertices]),
    );
    render_mesh.set_indices(Some(Indices::U32(buffer.indices.clone())));

    commands.spawn_bundle(LightBundle {
        transform: Transform::from_translation(Vec3::new(50.0, 50.0, 50.0)),
        light: Light {
            range: 200.0,
            intensity: 8000.0,
            ..Default::default()
        },
        ..Default::default()
    });
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 100.0))
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(render_mesh),
        material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
        transform: Transform::from_translation(Vec3::new(-32.0, -32.0, -32.0)),
        ..Default::default()
    });

    write_mesh_to_obj_file(&buffer);
}

fn write_mesh_to_obj_file(buffer: &SurfaceNetsBuffer) {
    export_to_file(
        &ObjSet {
            material_library: None,
            objects: vec![Object {
                name: "mesh".to_string(),
                vertices: buffer
                    .positions
                    .iter()
                    .map(|&[x, y, z]| Vertex {
                        x: x as f64,
                        y: y as f64,
                        z: z as f64,
                    })
                    .collect(),
                normals: buffer
                    .normals
                    .iter()
                    .map(|&[x, y, z]| Vertex {
                        x: x as f64,
                        y: y as f64,
                        z: z as f64,
                    })
                    .collect(),
                geometry: vec![Geometry {
                    material_name: None,
                    shapes: buffer
                        .indices
                        .chunks(3)
                        .map(|tri| Shape {
                            primitive: Primitive::Triangle(
                                (tri[0] as usize, None, Some(tri[0] as usize)),
                                (tri[1] as usize, None, Some(tri[1] as usize)),
                                (tri[2] as usize, None, Some(tri[2] as usize)),
                            ),
                            groups: vec![],
                            smoothing_groups: vec![],
                        })
                        .collect(),
                }],
                tex_vertices: vec![],
            }],
        },
        "mesh.obj",
    )
    .unwrap();
}

fn into_domain(array_dim: u32, [x, y, z]: [u32; 3]) -> [f32; 3] {
    [
        (2.0 * x as f32 / array_dim as f32) - 1.0,
        (2.0 * y as f32 / array_dim as f32) - 1.0,
        (2.0 * z as f32 / array_dim as f32) - 1.0,
    ]
}

fn sdf([x, y, z]: [f32; 3]) -> f32 {
    // sphere
    // (x * x + y * y + z * z).sqrt() - 0.9

    // link
    let le = 0.26;
    let r1 = 0.4;
    let r2 = 0.18;
    let [qx, qy, qz] = [x, (y.abs() - le).max(0.0), z];
    length2([length2([qx, qy]) - r1, qz]) - r2
}

fn length2([x, y]: [f32; 2]) -> f32 {
    (x * x + y * y).sqrt()
}

type SampleShape = ConstShape3u32<66, 66, 66>;
