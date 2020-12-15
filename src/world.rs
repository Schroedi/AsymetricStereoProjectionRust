use glium::{Display, VertexBuffer, IndexBuffer, Program};
use glium::index::PrimitiveType;
use cgmath::Vector3;

#[path = "teapot.rs"]
mod teapot;
#[path = "shaders.rs"]
mod shaders;

pub(crate) struct DisplayCorners {
    pub(crate) pa: Vector3<f32>,
    pub(crate) pb: Vector3<f32>,
    pub(crate) pc: Vector3<f32>,
}

pub(crate) struct Model {
    pub(crate) positions: VertexBuffer<teapot::Vertex>,
    pub(crate) normals: VertexBuffer<teapot::Normal>,
    pub(crate) indices: IndexBuffer<u16>,
    pub(crate) model_mat: [[f32; 4]; 4],
}

pub(crate) struct World {
    pub(crate) display_corners: DisplayCorners,
    pub(crate) models: Vec<Model>,
    pub(crate) shader: Program,
}

pub(crate) fn build_world(display: &Display) -> World {
    let model_teapot = Model {
        model_mat: [
            [0.0051, 0.0, 0.0, 0.0],
            [0.0, 0.0051, 0.0, 0.0],
            [0.0, 0.0, 0.0051, 0.0],
            [0.0, -0.450, -0.1, 1.0f32],
        ],
        positions: VertexBuffer::new(display, &teapot::VERTICES).unwrap(),
        normals: VertexBuffer::new(display, &teapot::NORMALS).unwrap(),
        indices: IndexBuffer::new(display, PrimitiveType::TrianglesList, &teapot::INDICES).unwrap(),
    };

    World {
        display_corners: DisplayCorners {
            pa: Vector3 {
                x: -0.41,
                y: -0.45,
                z: 0.0,
            },
            pb: Vector3 {
                x: 0.41,
                y: -0.45,
                z: 0.0,
            },
            pc: Vector3 {
                x: -0.41,
                y: 0.00,
                z: 0.0,
            },
        },
        models: Vec::from([model_teapot]),
        shader: shaders::build_shader(&display),
    }
}
