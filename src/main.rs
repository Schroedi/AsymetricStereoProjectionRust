#![feature(array_map)]

extern crate open_track;

use std::fmt;
use std::time::Instant;

use cgmath::{Matrix3, Matrix4, Vector4, Point3, Perspective, PerspectiveFov, Rad, Deg};
use cgmath::prelude::*;
use cgmath::Vector3;
use glium::{Display, Frame, IndexBuffer, Program, Rect, Surface, uniform, VertexBuffer, DrawParameters};
use glium::index::PrimitiveType;
use glutin::window::WindowBuilder;
use imgui_glium_renderer::imgui::{Context, FontConfig, FontSource, im_str, Ui};
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::event::{Event, WindowEvent, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit_input_helper::WinitInputHelper;

use open_track::OpenTrackServer;

#[path = "teapot.rs"]
mod teapot;

#[allow(dead_code)]
#[derive(Debug)]
enum Eyes {
    LEFT,
    RIGHT,
    CYCLOPS,
}

#[allow(dead_code)]
enum StereoModes {
    ANAGLYPH,
    SBS,
}

#[allow(dead_code)]
enum TrackerModes {
    OPENTRACK,
    VRPN,
    CONSTANT,
}

struct DisplayCorners {
    pa: Vector3<f32>,
    pb: Vector3<f32>,
    pc: Vector3<f32>,
}

struct Model {
    positions: VertexBuffer<teapot::Vertex>,
    normals: VertexBuffer<teapot::Normal>,
    indices: IndexBuffer<u16>,
    model_mat: [[f32; 4]; 4],
}

struct World {
    display_corners: DisplayCorners,
    models: Vec<Model>,
    shader: Program,
}

const EYE_SEPARATION: f32 =0.06;
const STEREO_MODE: StereoModes = StereoModes::SBS;
const TRACKER_MODE: TrackerModes = TrackerModes::OPENTRACK;

fn vec_from_array3(v : &[f32; 3]) -> Vector3<f32> {
    return Vector3::new(v[0], v[1], v[2]);
}

fn vec_from_array4(v : &[f32; 4]) -> Vector4<f32> {
    return Vector4::new(v[0], v[1], v[2], v[3]);
}

fn print_mat(m : &Matrix4<f32>) -> String {
    let mut res : String = "".to_string();
    res += &*format!("\n{:?}", m.row(0));
    res += &*format!("\n{:?}", m.row(1));
    res += &*format!("\n{:?}", m.row(2));
    res += &*format!("\n{:?}", m.row(3));
    return res;
}

fn print_mat_ui(m : &Matrix4<f32>, ui: &mut Ui) {
    ui.text(im_str!("{:+.3?}", m.row(0)));
    ui.text(im_str!("{:+.3?}", m.row(1)));
    ui.text(im_str!("{:+.3?}", m.row(2)));
    ui.text(im_str!("{:+.3?}", m.row(3)));
}

fn main() {
    let title = "Stereo testing";
    let event_loop = EventLoop::new();
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let builder = WindowBuilder::new()
        .with_title(title.to_owned())
        .with_inner_size(glutin::dpi::LogicalSize::new(1920_f64, 1080_f64));
    let display =
        Display::new(builder, context, &event_loop).expect("Failed to initialize display");

    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    let mut platform = WinitPlatform::init(&mut imgui);
    {
        let gl_window = display.gl_window();
        let window = gl_window.window();
        platform.attach_window(imgui.io_mut(), window, HiDpiMode::Rounded);
    }

    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[FontSource::DefaultFontData {
        config: Some(FontConfig {
            size_pixels: font_size,
            ..FontConfig::default()
        }),
    }]);
    imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

    let mut renderer = Renderer::init(&mut imgui, &display).expect("Failed to initialize renderer");

    // our stuff
    let world = build_world(&display);
    let tracker = open_track::OpenTrackServer::start(None);

    let mut input = WinitInputHelper::new();
    let mut last_frame = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        if input.update(&event) {
            if input.key_released(VirtualKeyCode::Q) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }
        match event {
        Event::NewEvents(_) => {
            let now = Instant::now();
            imgui.io_mut().update_delta_time(now - last_frame);
            last_frame = now;
        }
        Event::MainEventsCleared => {
            let gl_window = display.gl_window();
            platform
                .prepare_frame(imgui.io_mut(), gl_window.window())
                .expect("Failed to prepare frame");
            gl_window.window().request_redraw();
        }
        Event::RedrawRequested(_) => {
            let mut ui = imgui.frame();
            let gl_window = display.gl_window();
            let mut target = display.draw();
            target.clear_color_and_depth((0.1, 0.1, 0.11, 1.0), 1.0);

            render3d(&mut target, &world, &tracker, &mut ui);

            // render UI
            platform.prepare_render(&ui, gl_window.window());
            renderer
                .render(&mut target, ui.render())
                .expect("Rendering failed");

            target.finish().expect("Failed to swap buffers");
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *control_flow = ControlFlow::Exit,
        event => {
            let gl_window = display.gl_window();
            platform.handle_event(imgui.io_mut(), gl_window.window(), &event);
        }
    }})
}

fn build_shader(display: &Display) -> Program {
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
            //color = vec4(1.0, 1.0, 0.0, 1.0);
        }
    "#;
    Program::from_source(display, vertex_shader_src, fragment_shader_src, None).unwrap()
}

fn build_world(display: &Display) -> World {
    let model_teapot = Model {
        model_mat: [
            [0.01, 0.0, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
            [0.0, 0.0, -10.0, 1.0f32],
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
        shader: build_shader(&display),
    }
}

fn render3d(target: &mut Frame, world: &World, tracker: &OpenTrackServer, ui: &mut Ui) {
    // debugging info about the world
    {
        let width = (world.display_corners.pb - world.display_corners.pa).magnitude();
        let height = (world.display_corners.pc - world.display_corners.pa).magnitude();
        ui.text(im_str!("Screen size {:?}", [width, height]));
    }

    let eyes = [Eyes::LEFT, Eyes::RIGHT];
    for eye in eyes.iter() {
        ui.separator();
        ui.text(im_str!("{:?} eye", eye));

        // Eye position
        let pos = eye_pos(tracker, eye);
        ui.text(im_str!("pos {:+.3?}", pos));

        let viewport_rect = viewport(target, eye);
        // let perspective = simple_projection(viewport_rect.unwrap().width,
        //                                     viewport_rect.unwrap().height, &pos,
        //                                     0.01, 1000.0);

        let perspective = general_projection(&world.display_corners, &pos, 0.01, 1000.0);
        ui.text(im_str!("projection:"));
        print_mat_ui(&perspective, ui);

        let params = draw_parameters(eye, viewport_rect);

        // uniforms
        let light = [1.4, 0.4, 0.7f32];
        let view_mat : [[f32; 4]; 4] = *Matrix4::identity().as_ref();
        let cam_projection: [[f32; 4]; 4] = *perspective.as_ref();

        for m in world.models.as_slice() {
            let uniforms = &uniform! {
                model: m.model_mat,
                view: view_mat,
                perspective: cam_projection,
                u_light: light };
            target.draw((&m.positions, &m.normals), &m.indices,
                        &world.shader, uniforms,
                        &params).unwrap();
        }
    } // eye
}

fn draw_parameters(eye: &Eyes, viewport_rect: Option<Rect>) -> DrawParameters {
    let params = glium::DrawParameters {
        color_mask: match STEREO_MODE {
            StereoModes::ANAGLYPH => match eye {
                Eyes::LEFT => (true, false, false, true),
                Eyes::RIGHT => (false, true, true, true),
                Eyes::CYCLOPS => (true, true, true, true),
            },
            _ => (true, true, true, true),
        },
        viewport: viewport_rect,
        depth: glium::Depth {
            test: glium::draw_parameters::DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
        ..Default::default()
    };
    params
}

fn viewport(target: &mut Frame, eye: &Eyes) -> Option<Rect> {
    let (window_width, window_height) = target.get_dimensions();
    let viewport_rect = match eye {
        Eyes::LEFT => Some(Rect {
            left: 0,
            bottom: 0,
            width: window_width / 2,
            height: window_height,
        }),
        Eyes::RIGHT => Some(Rect {
            left: window_width / 2,
            bottom: 0,
            width: window_width / 2,
            height: window_height,
        }),
        Eyes::CYCLOPS => Some(Rect {
            left: 0,
            bottom: 0,
            width: window_width,
            height: window_height,
        }),
    };
    viewport_rect
}

fn eye_pos(tracker: &OpenTrackServer, eye: &Eyes) -> Vector3<f32> {
    let eye_offset = match eye {
        Eyes::LEFT => -EYE_SEPARATION / 2.0,
        Eyes::RIGHT => EYE_SEPARATION / 2.0,
        _ => 0.0,
    };
    let mut pos: Vector3<f32> = Vector3::zero();
    let mut _rot: Vector3<f32> = Vector3::unit_z();
    match TRACKER_MODE {
        TrackerModes::OPENTRACK => {
            let (p, _r) = match tracker.get_pos_rot() {
                Some(v) => (vec_from_array3(&v.0) / 100.0, vec_from_array3(&v.1)),
                None => (Vector3::zero(), Vector3::unit_z()),
            };
            pos = p;
        }
        TrackerModes::VRPN => {}
        TrackerModes::CONSTANT => {
            pos = Vector3 {x: 0.0, y: 0.0, z: 0.6}
        }
    }

    // TODO: include hear orientation
    pos += eye_offset * Vector3::unit_x();
    pos
}

fn simple_projection(window_width: u32, window_height: u32, pe: &Vector3<f32>, near: f32, far: f32,)
    -> Matrix4<f32> {
    let T = Matrix4::from_translation(-*pe);
    let V = Matrix4::look_at(Point3::from_vec(*pe), Point3::origin(), Vector3::unit_y());
    let width = match STEREO_MODE {
        StereoModes::SBS => window_width as f32 / 2.0,
        _ => window_width as f32,
    };
    let aspect_ratio = window_height as f32 / width;
    let pers = PerspectiveFov {
        fovy: Rad::from(Deg(75_f32)),
        aspect: aspect_ratio,
        near,
        far
    };
    Matrix4::from(pers) * V * T
}

fn general_projection(
    display: &DisplayCorners,
    pe: &Vector3<f32>,
    near: f32,
    far: f32,
) -> Matrix4<f32> {
    let (pa, pb, pc) = (display.pa, display.pb, display.pc);

    let vr: Vector3<f32> = (pb - pa).normalize();
    let vu: Vector3<f32> = (pc - pa).normalize();
    let vn = (vr.cross(vu)).normalize();

    let va = pa - pe;
    let vb = pb - pe;
    let vc = pc - pe;

    let d = -(vn.dot(va));
    assert!(d >= 0.0001, "eye-screen distance is zero or less => this would mean that the eye is inside the \
    monitor. Is the tracking working and the coordinate system in the right direction?");

    let nd = near / d;
    let l = (vr.dot(va)) * nd;
    let b = (vu.dot(va)) * nd;
    let r = (vr.dot(vb)) * nd;
    let t = (vu.dot(vc)) * nd;
    let P = cgmath::frustum(l, r, b, t, near, far);

    let Mt = Matrix4::from(Matrix3::from_cols(vr, vu, vn).transpose());

    let T = Matrix4::from_translation(-*pe);

    P * Mt * T
    //P * T
}
