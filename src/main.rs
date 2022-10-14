extern crate open_track;
mod mat_helpers;
mod world;

use std::time::Instant;

use cgmath::{Matrix3, Matrix4, Point3, PerspectiveFov, Rad, Deg};
use cgmath::prelude::*;
use cgmath::Vector3;
use glium::{Display, Frame, Rect, Surface, uniform, DrawParameters};
use glutin::window::WindowBuilder;
use imgui_glium_renderer::imgui::{Context, FontConfig, FontSource, Ui};
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::event::{Event, WindowEvent, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit_input_helper::WinitInputHelper;

use open_track::OpenTrackServer;
use crate::mat_helpers::{print_mat_ui, vec_from_array3};
use crate::world::{World, DisplayCorners};


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
    PANCAKE,
}

#[allow(dead_code)]
enum TrackerModes {
    OPENTRACK,
    VRPN,
    CONSTANT,
}


/// Config
const EYE_SEPARATION: f32 = 0.06;
const STEREO_MODE: StereoModes = StereoModes::PANCAKE;
const TRACKER_MODE: TrackerModes = TrackerModes::OPENTRACK;

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
    let world = world::build_world(&display);
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

fn render3d(target: &mut Frame, world: &World, tracker: &OpenTrackServer, ui: &mut Ui) {
    // debugging info about the world
    {
        let width = (world.display_corners.pb - world.display_corners.pa).magnitude();
        let height = (world.display_corners.pc - world.display_corners.pa).magnitude();
        ui.text(format!("Screen size {:?}", [width, height]));
    }

    match STEREO_MODE {
        StereoModes::PANCAKE =>
            reder_for_eye(target, &world, tracker, ui, &Eyes::CYCLOPS),
        _ => {
            reder_for_eye(target, &world, tracker, ui, &Eyes::LEFT);
            reder_for_eye(target, &world, tracker, ui, &Eyes::RIGHT);
        }
    };
}

fn reder_for_eye(target: &mut Frame, world: &&World, tracker: &OpenTrackServer, ui: &mut Ui, eye: &Eyes) {
    ui.separator();
    ui.text(format!("{:?} eye", eye));

    // Eye position
    let pos = eye_pos(tracker, eye);
    ui.text(format!("pos {:+.3?}", pos));

    let viewport_rect = viewport(target, eye);
    // let perspective = simple_projection(viewport_rect.unwrap().width,
    //                                     viewport_rect.unwrap().height, &pos,
    //                                     0.01, 1000.0);

    let perspective = general_projection(&world.display_corners, &pos, 0.1, 1000.0, ui);
    ui.text(format!("projection:"));
    print_mat_ui(&perspective, ui);

    let params = draw_parameters(eye, viewport_rect);

    // uniforms
    let light = [1.4, 0.4, 0.7f32];
    let view_mat: [[f32; 4]; 4] = *Matrix4::identity().as_ref();
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
        //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullingDisabled,
        ..Default::default()
    };
    params
}

fn viewport(target: &mut Frame, eye: &Eyes) -> Option<Rect> {
    let (window_width, window_height) = target.get_dimensions();
    match STEREO_MODE {
        StereoModes::SBS => match eye {
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
        },
        _ => Some(Rect {
            left: 0,
            bottom: 0,
            width: window_width,
            height: window_height,
        }),
    }
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
    let V = Matrix4::look_at_rh(Point3::from_vec(*pe), Point3::origin(), Vector3::unit_y());
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
    ui: &mut Ui
) -> Matrix4<f32> {
    let (pa, pb, pc) = (display.pa, display.pb, display.pc);

    let vr: Vector3<f32> = (pb - pa).normalize();
    let vu: Vector3<f32> = (pc - pa).normalize();
    let vn = (vr.cross(vu)).normalize();

    let va = pa - pe;
    let vb = pb - pe;
    let vc = pc - pe;

    let mut d = -(va.dot(vn));
    if d < 0.0001 {
        eprintln!("eye-screen distance is zero or less => this would mean that the eye is inside the \
    monitor. Is the tracking working and the coordinate system in the right direction?");
        d = 0.0001;
    }

    let nd = near / d;
    let l = (vr.dot(va)) * nd;
    let r = (vr.dot(vb)) * nd;
    let b = (vu.dot(va)) * nd;
    let t = (vu.dot(vc)) * nd;
    let P = cgmath::frustum(l, r, b, t, near, far);

    // this is identity for a display aligned with the tracking space
    let Mt = Matrix4::from(Matrix3::from_cols(vr, vu, vn).transpose());

    let T = Matrix4::from_translation(-*pe);

    ui.text(format!("general projection"));
    ui.text(format!("l {:.2?}, r {:.2?}, b {:.2?}, t {:.2?}:", l, r, b, t));
    ui.text(format!("P:"));
    print_mat_ui(&P, ui);
    ui.text(format!("T:"));
    print_mat_ui(&T, ui);

    let comb = P * Mt * T;
    comb
}
