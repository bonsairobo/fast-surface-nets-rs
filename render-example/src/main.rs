use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};

use bevy::{
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
};

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
        samples[i as usize] = sphere_sdf(p);
    }

    // Do a single run first to allocate the buffer to the right size.
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&samples, &SampleShape {}, [0; 3], [65; 3], &mut buffer);

    let num_vertices = buffer.positions.len();

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float3(buffer.positions),
    );
    render_mesh.set_attribute(
        "Vertex_Normal",
        VertexAttributeValues::Float3(buffer.normals),
    );
    render_mesh.set_attribute(
        "Vertex_Uv",
        VertexAttributeValues::Float2(vec![[0.0; 2]; num_vertices]),
    );
    render_mesh.set_indices(Some(Indices::U32(buffer.indices)));

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
        transform: Transform::from_translation(Vec3::new(0.0, 0.0, 150.0))
            .looking_at(Vec3::new(0.0, 10.0, 0.0), Vec3::Y),
        ..Default::default()
    });
    commands.spawn_bundle(PbrBundle {
        mesh: meshes.add(render_mesh),
        material: materials.add(Color::rgb(1.0, 0.0, 0.0).into()),
        transform: Transform::from_translation(Vec3::new(-32.0, -32.0, -32.0)),
        ..Default::default()
    });
}

fn into_domain(array_dim: u32, [x, y, z]: [u32; 3]) -> [f32; 3] {
    [
        (2.0 * x as f32 / array_dim as f32) - 1.0,
        (2.0 * y as f32 / array_dim as f32) - 1.0,
        (2.0 * z as f32 / array_dim as f32) - 1.0,
    ]
}

fn sphere_sdf([x, y, z]: [f32; 3]) -> f32 {
    (x * x + y * y + z * z) - 0.9
}

type SampleShape = ConstShape3u32<66, 66, 66>;
