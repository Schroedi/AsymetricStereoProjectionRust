#![feature(array_map)]

#[macro_use]
extern crate glium;

use glium::Rect;

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

const STEREO_MODE: StereoModes = StereoModes::SBS;

fn main() {
    #![allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new().with_depth_buffer(24);
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

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

    let tracker = open_track::OpenTrackServer::start(None);

    event_loop.run(move |event, _, control_flow| {
        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
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
        for eye in eyes.iter() {

            let eye_offset = match eye {
                Eyes::LEFT => -0.06,
                Eyes::RIGHT => 0.06,
                _ => {0.0}
            };
            //let pos = [0.0 + eye_offset, 0.0, 60.0];
            let (pos, _rot) = tracker.get_pos_rot();

            let view = view_matrix(&pos, &[0.0, 0.0, -1.0], &[0.0, 1.0, 0.0]);

            let (window_width, window_height) = target.get_dimensions();
            let perspective = {
                let aspect_ratio = window_height as f32 / match STEREO_MODE {
                    StereoModes::SBS => window_width as f32 / 2.0,
                    _ => window_width as f32,
                } as f32;

                let fov: f32 = std::f32::consts::PI / 3.0;
                let zfar = 1024.0;
                let znear = 0.1;

                let f = 1.0 / (fov / 2.0).tan();

                [
                    [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                    [         0.0         ,     f ,              0.0              ,   0.0],
                    [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                    [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
                ]
            };

            let light = [1.4, 0.4, -0.7f32];

            let params = glium::DrawParameters {
                color_mask:
                match STEREO_MODE {
                    StereoModes::ANAGLYPH =>
                        match eye {
                        Eyes::LEFT => (true, false, false, true),
                        Eyes::RIGHT => (false, true, true, true),
                        Eyes::CYCLOPS => (true, true, true, true),
                        },
                    _ =>
                        (true, true, true, true)
                },
                viewport:
                match STEREO_MODE {
                    StereoModes::ANAGLYPH => None,
                    StereoModes::SBS => match eye {
                        Eyes::LEFT => Some(Rect{left: 0, bottom:0, width: window_width /2, height:768}),
                        Eyes::RIGHT => Some(Rect{left: window_width /2, bottom:0, width: window_width /2, height:768}),
                        Eyes::CYCLOPS => None,
                    },
                },
                depth: glium::Depth {
                    test: glium::draw_parameters::DepthTest::IfLess,
                    write: true,
                    .. Default::default()
                },
                //backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockWise,
                .. Default::default()
            };

            target.draw((&positions, &normals), &indices, &program,
                        &uniform! { model: model, view: view, perspective: perspective, u_light: light },
                        &params).unwrap();
        }


        target.finish().unwrap();
    });
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