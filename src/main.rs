#![feature(array_map)]

#[macro_use]
extern crate glium;

use glium::Rect;

use cgmath::prelude::*;
use cgmath::{Matrix4, Matrix3};
use cgmath::Vector3;
use imgui_glium_renderer::imgui::{Context, Ui, FontSource, FontConfig};
use imgui_glium_renderer::imgui::im_str;
use std::time::Instant;
use glium::glutin::event::Event;
use imgui_glium_renderer::Renderer;

#[path = "teapot.rs"]
mod teapot;

extern crate open_track;

enum Eyes {
    LEFT, RIGHT, CYCLOPS
}

#[allow(dead_code)]
enum StereoModes {
    ANAGLYPH, SBS
}

struct DisplayCorners {
    pa: Vector3<f32>,
    pb: Vector3<f32>,
    pc: Vector3<f32>,
}

const STEREO_MODE: StereoModes = StereoModes::SBS;

fn main() {
    #![allow(unused_imports)]
    use glium::{glutin, Surface};

    let display_corners = DisplayCorners {
        pa: Vector3 {x: -0.41, y: -0.45, z: 0.0},
        pb: Vector3 {x:  0.41, y: -0.45, z: 0.0},
        pc: Vector3 {x: -0.41, y:  0.00, z: 0.0}, };

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Projection")
        .with_inner_size(glutin::dpi::LogicalSize::new(1920_f64, 1024_f64));
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let mut imgui = Context::create();
    imgui.set_ini_filename(None);
    imgui.io_mut().display_size = [1920_f32, 1024_f32];
    let mut renderer = Renderer::init(&mut imgui, &display).expect("Failed to initialize renderer");
    let hidpi_factor = 2.0;
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        },
        // FontSource::TtfData {
        //     data: include_bytes!("../../../resources/mplus-1p-regular.ttf"),
        //     size_pixels: font_size,
        //     config: Some(FontConfig {
        //         rasterizer_multiply: 1.75,
        //         glyph_ranges: FontGlyphRanges::japanese(),
        //         ..FontConfig::default()
        //     }),
        // },
    ]);

    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    let positions = glium::VertexBuffer::new(&display, &teapot::VERTICES).unwrap();
    let normals = glium::VertexBuffer::new(&display, &teapot::NORMALS).unwrap();
    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList,
                                          &teapot::INDICES).unwrap();

    let vertex_shader_src = r#"
        #version 150
        in vec3 position;
        in vec3 normal;
        out vec3 v_normal;
        out vec3 v_position;
        uniform mat4 perspective;
        uniform mat4 view;
        uniform mat4 model;
        void main() {
            mat4 modelview = view * model;
            v_normal = transpose(inverse(mat3(modelview))) * normal;
            gl_Position = perspective * modelview * vec4(position, 1.0);
            v_position = gl_Position.xyz / gl_Position.w;
        }
    "#;

    let fragment_shader_src = r#"
        #version 150
        in vec3 v_normal;
        in vec3 v_position;
        out vec4 color;
        uniform vec3 u_light;
        const vec3 ambient_color = vec3(0.2, 0.0, 0.0);
        const vec3 diffuse_color = vec3(0.6, 0.0, 0.0);
        const vec3 specular_color = vec3(1.0, 1.0, 1.0);
        void main() {
            float diffuse = max(dot(normalize(v_normal), normalize(u_light)), 0.0);
            vec3 camera_dir = normalize(-v_position);
            vec3 half_direction = normalize(normalize(u_light) + camera_dir);
            float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 16.0);
            color = vec4(ambient_color + diffuse * diffuse_color + specular * specular_color, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src,
                                              None).unwrap();

    //let tracker = open_track::OpenTrackServer::start(None);

    let mut last_frame = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        match event {
            Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let now = Instant::now();
        imgui.io_mut().update_delta_time(now - last_frame);
        last_frame = now;
        imgui.pre

        let mut ui = imgui.frame();

        let mut run = true;
        run_ui(&mut run, &mut ui);
        if !run {
            *control_flow = glutin::event_loop::ControlFlow::Exit;
        }

        let mut target = display.draw();
        target.clear_color_and_depth((0.1, 0.1, 0.11, 1.0), 1.0);

        let model = [
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.0],
            [0.0, 0.0, 0.0, 1.0f32]
        ];

        let eyes = [Eyes::LEFT, Eyes::RIGHT];
        // for eye in eyes.iter() {
        //
        //     let eye_offset = match eye {
        //         Eyes::LEFT => -0.06,
        //         Eyes::RIGHT => 0.06,
        //         _ => {0.0}
        //     };
        //     //let pos = [0.0 + eye_offset, 0.0, 60.0];
        //     let (pos, _rot) = tracker.get_pos_rot();
        //
        //     let view = view_matrix(&pos, &[0.0, 0.0, -1.0], &[0.0, 1.0, 0.0]);
        //     let (window_width, window_height) = target.get_dimensions();
        //     let perspective = perspective_projection(window_width, window_height);
        //
        //     //let perspective = general_projection(&display_corners, &pos, 0.01, 1000.0);
        //
        //     let light = [1.4, 0.4, -0.7f32];
        //
        //     let params = glium::DrawParameters {
        //         color_mask:
        //         match STEREO_MODE {
        //             StereoModes::ANAGLYPH =>
        //                 match eye {
        //                 Eyes::LEFT => (true, false, false, true),
        //                 Eyes::RIGHT => (false, true, true, true),
        //                 Eyes::CYCLOPS => (true, true, true, true),
        //                 },
        //             _ =>
        //                 (true, true, true, true)
        //         },
        //         viewport:
        //         match STEREO_MODE {
        //             StereoModes::ANAGLYPH => None,
        //             StereoModes::SBS => match eye {
        //                 Eyes::LEFT => Some(Rect{left: 0, bottom:0, width: window_width /2, height:768}),
        //                 Eyes::RIGHT => Some(Rect{left: window_width /2, bottom:0, width: window_width /2, height:768}),
        //                 Eyes::CYCLOPS => None,
        //             },
        //         },
        //         depth: glium::Depth {
        //             test: glium::draw_parameters::DepthTest::IfLess,
        //             write: true,
        //             .. Default::default()
        //         },
        //         //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockWise,
        //         .. Default::default()
        //     };
        //
        //     target.draw((&positions, &normals), &indices, &program,
        //                 &uniform! { model: model, view: view, perspective: perspective, u_light: light },
        //                 &params).unwrap();
        // } // eye

        let draw_data = ui.render();
        renderer
            .render(&mut target, draw_data)
            .expect("Rendering failed");
        target.finish().unwrap();
    });
}

fn run_ui(run: &mut bool, ui: &mut Ui) {
    ui.text(im_str!("Hello world!"));
    //ui.text(im_str!("こんにちは世界！"));
    ui.text(im_str!("This...is...imgui-rs!"));
    ui.separator();
    let mouse_pos = ui.io().mouse_pos;
    println!("{:?}", mouse_pos);
    ui.text(format!(
        "Mouse Position: ({:.1},{:.1})",
        mouse_pos[0], mouse_pos[1]
    ));
}


fn general_projection(display: &DisplayCorners, pe: &Vector3<f32>, near: f32, far: f32) -> Matrix4<f32> {
    let (pa, pb, pc) = (display.pa, display.pb, display.pc);

    let vr : Vector3<f32> = (pb - pa).normalize();
    let vu : Vector3<f32> = (pc - pa).normalize();
    let vn = (vr.cross(vu)).normalize();

    let va = pa - pe;
    let vb = pb - pe;
    let vc = pc - pe;

    let d = -(vn.dot(va));

    let l = (vr.dot(va)) * near/d;
    let r = (vr.dot(vb)) * near/d;
    let b = (vu.dot(va)) * near/d;
    let t = (vu.dot(vc)) * near/d;
    let P = cgmath::frustum(l, r, b, t, near, far);

    let Mt = Matrix4::from(Matrix3::from_cols(vr, vu, vn).transpose());

    let T = Matrix4::from_translation(-*pe);

    P * Mt * T
}

fn perspective_projection(window_width: u32, window_height: u32) -> [[f32; 4]; 4]{
    let aspect_ratio = window_height as f32 / match STEREO_MODE {
        StereoModes::SBS => window_width as f32 / 2.0,
        _ => window_width as f32,
    } as f32;

    let fov: f32 = std::f32::consts::PI / 3.0;
    let zfar = 1024.0;
    let znear = 0.1;

    let f = 1.0 / (fov / 2.0).tan();

    [
        [f * aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (zfar + znear) / (zfar - znear), 1.0],
        [0.0, 0.0, -(2.0 * zfar * znear) / (zfar - znear), 0.0],
    ]
}

fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
        up[2] * f[0] - up[0] * f[2],
        up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
        f[2] * s_norm[0] - f[0] * s_norm[2],
        f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
        -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
        -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s_norm[0], u[0], f[0], 0.0],
        [s_norm[1], u[1], f[1], 0.0],
        [s_norm[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
    ]
}