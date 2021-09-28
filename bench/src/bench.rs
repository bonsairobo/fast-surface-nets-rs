use fast_surface_nets::ndshape::{ConstShape, ConstShape3u32};
use fast_surface_nets::{surface_nets, SignedDistance, SurfaceNetsBuffer};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::f32::consts::PI;

type SampleShape = ConstShape3u32<18, 18, 18>;

fn bench_empty_space(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_empty_space");
    let samples = [Sd8(i8::MAX); SampleShape::SIZE as usize];

    // Do a single run first to allocate the buffer to the right size.
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&samples, &SampleShape {}, [0; 3], [17; 3], &mut buffer);
    let num_triangles = buffer.indices.len() / 3;

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("tris={}", num_triangles)),
        &(),
        |b, _| {
            b.iter(|| surface_nets(&samples, &SampleShape {}, [0; 3], [17; 3], &mut buffer));
        },
    );
    group.finish();
}

fn bench_sine_sdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_sine_sdf");
    let mut samples = [Sd8(i8::MAX); SampleShape::SIZE as usize];
    for i in 0u32..(SampleShape::SIZE) {
        let p = into_domain(16, SampleShape::delinearize(i));
        samples[i as usize] = sine_sdf(5.0, p);
    }

    // Do a single run first to allocate the buffer to the right size.
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&samples, &SampleShape {}, [0; 3], [17; 3], &mut buffer);
    let num_triangles = buffer.indices.len() / 3;

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("tris={}", num_triangles)),
        &(),
        |b, _| {
            b.iter(|| surface_nets(&samples, &SampleShape {}, [0; 3], [17; 3], &mut buffer));
        },
    );
    group.finish();
}

fn bench_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("bench_sphere");
    let mut samples = [Sd8(i8::MAX); SampleShape::SIZE as usize];
    for i in 0u32..(SampleShape::SIZE) {
        let p = into_domain(16, SampleShape::delinearize(i));
        samples[i as usize] = sphere_sdf(p);
    }

    // Do a single run first to allocate the buffer to the right size.
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&samples, &SampleShape {}, [0; 3], [17; 3], &mut buffer);
    let num_triangles = buffer.indices.len() / 3;

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("tris={}", num_triangles)),
        &(),
        |b, _| {
            b.iter(|| surface_nets(&samples, &SampleShape {}, [0; 3], [17; 3], &mut buffer));
        },
    );
    group.finish();
}

criterion_group!(benches, bench_sine_sdf, bench_sphere, bench_empty_space);
criterion_main!(benches);

// The higher the frequency (n) the more surface area to mesh.
fn sine_sdf(n: f32, [x, y, z]: [f32; 3]) -> Sd8 {
    let val = ((x / 2.0) * n * PI).sin() + ((y / 2.0) * n * PI).sin() + ((z / 2.0) * n * PI).sin();

    val.into()
}

fn sphere_sdf([x, y, z]: [f32; 3]) -> Sd8 {
    let val = (x * x + y * y + z * z) - 0.9;

    val.into()
}

fn into_domain(array_dim: u32, [x, y, z]: [u32; 3]) -> [f32; 3] {
    [
        (2.0 * x as f32 / array_dim as f32) - 1.0,
        (2.0 * y as f32 / array_dim as f32) - 1.0,
        (2.0 * z as f32 / array_dim as f32) - 1.0,
    ]
}

#[derive(Clone, Copy)]
struct Sd8(pub i8);

impl Sd8 {
    const RESOLUTION: f32 = i8::MAX as f32;
    const PRECISION: f32 = 1.0 / Self::RESOLUTION;
}

impl From<Sd8> for f32 {
    fn from(d: Sd8) -> Self {
        d.0 as f32 * Sd8::PRECISION
    }
}

impl From<f32> for Sd8 {
    fn from(d: f32) -> Self {
        Self((Self::RESOLUTION * d.min(1.0).max(-1.0)) as i8)
    }
}

impl SignedDistance for Sd8 {
    fn is_negative(self) -> bool {
        self.0 < 0
    }
}
