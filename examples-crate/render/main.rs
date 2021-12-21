use fast_surface_nets::glam::{Vec2, Vec3A};
use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};

use bevy::{
    pbr::wireframe::{WireframeConfig, WireframePlugin},
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        options::WgpuOptions,
        render_resource::{PrimitiveTopology, WgpuFeatures},
    },
};
use obj_exporter::{export_to_file, Geometry, ObjSet, Object, Primitive, Shape, Vertex};

fn main() {
    App::new()
        .insert_resource(WgpuOptions {
            features: WgpuFeatures::POLYGON_MODE_LINE,
            ..Default::default()
        })
        .insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins)
        .add_plugin(WireframePlugin)
        .add_startup_system(setup.system())
        .run();
}

fn setup(
    mut commands: Commands,
    mut wireframe_config: ResMut<WireframeConfig>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    wireframe_config.global = true;

    commands.spawn_bundle(PointLightBundle {
        transform: Transform::from_translation(Vec3::new(25.0, 25.0, 25.0)),
        point_light: PointLight {
            range: 200.0,
            intensity: 8000.0,
            ..Default::default()
        },
        ..Default::default()
    });
    commands.spawn_bundle(PerspectiveCameraBundle {
        transform: Transform::from_translation(Vec3::new(50.0, 15.0, 50.0))
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        ..Default::default()
    });

    let (sphere_buffer, sphere_mesh) = sdf_to_mesh(&mut meshes, |p| sphere(0.9, p));
    let (cube_buffer, cube_mesh) = sdf_to_mesh(&mut meshes, |p| cube(Vec3A::splat(0.5), p));
    let (link_buffer, link_mesh) = sdf_to_mesh(&mut meshes, |p| link(0.26, 0.4, 0.18, p));

    spawn_pbr(
        &mut commands,
        &mut materials,
        sphere_mesh,
        Transform::from_translation(Vec3::new(-16.0, -16.0, -16.0)),
    );
    spawn_pbr(
        &mut commands,
        &mut materials,
        cube_mesh,
        Transform::from_translation(Vec3::new(-16.0, -16.0, 16.0)),
    );
    spawn_pbr(
        &mut commands,
        &mut materials,
        link_mesh,
        Transform::from_translation(Vec3::new(16.0, -16.0, -16.0)),
    );

    write_mesh_to_obj_file("sphere".into(), &sphere_buffer);
    write_mesh_to_obj_file("cube".into(), &cube_buffer);
    write_mesh_to_obj_file("link".into(), &link_buffer);
}

fn sdf_to_mesh(
    meshes: &mut Assets<Mesh>,
    sdf: impl Fn(Vec3A) -> f32,
) -> (SurfaceNetsBuffer, Handle<Mesh>) {
    type SampleShape = ConstShape3u32<34, 34, 34>;

    let mut samples = [1.0; SampleShape::SIZE as usize];
    for i in 0u32..(SampleShape::SIZE) {
        let p = into_domain(32, SampleShape::delinearize(i));
        samples[i as usize] = sdf(p);
    }

    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&samples, &SampleShape {}, [0; 3], [33; 3], &mut buffer);

    let num_vertices = buffer.positions.len();

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float32x3(buffer.positions.clone()),
    );
    render_mesh.set_attribute(
        "Vertex_Normal",
        VertexAttributeValues::Float32x3(buffer.normals.clone()),
    );
    render_mesh.set_attribute(
        "Vertex_Uv",
        VertexAttributeValues::Float32x2(vec![[0.0; 2]; num_vertices]),
    );
    render_mesh.set_indices(Some(Indices::U32(buffer.indices.clone())));

    (buffer, meshes.add(render_mesh))
}

fn spawn_pbr(
    commands: &mut Commands,
    materials: &mut Assets<StandardMaterial>,
    mesh: Handle<Mesh>,
    transform: Transform,
) {
    let mut material = StandardMaterial::from(Color::rgb(0.0, 0.0, 0.0));
    material.perceptual_roughness = 0.9;

    commands.spawn_bundle(PbrBundle {
        mesh,
        material: materials.add(material),
        transform,
        ..Default::default()
    });
}

fn write_mesh_to_obj_file(name: String, buffer: &SurfaceNetsBuffer) {
    let filename = format!("{}.obj", name);
    export_to_file(
        &ObjSet {
            material_library: None,
            objects: vec![Object {
                name,
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
        filename,
    )
    .unwrap();
}

fn into_domain(array_dim: u32, [x, y, z]: [u32; 3]) -> Vec3A {
    (2.0 / array_dim as f32) * Vec3A::new(x as f32, y as f32, z as f32) - 1.0
}

fn sphere(radius: f32, p: Vec3A) -> f32 {
    p.length() - radius
}

fn cube(b: Vec3A, p: Vec3A) -> f32 {
    let q = p.abs() - b;
    q.max(Vec3A::ZERO).length() + q.max_element().min(0.0)
}

fn link(le: f32, r1: f32, r2: f32, p: Vec3A) -> f32 {
    let q = Vec3A::new(p.x, (p.y.abs() - le).max(0.0), p.z);
    Vec2::new(q.length() - r1, q.z).length() - r2
}
