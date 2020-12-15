use cgmath::{Vector3, Vector4, Matrix4, Matrix};
use imgui_glium_renderer::imgui::{im_str, Ui};

pub fn vec_from_array3(v : &[f32; 3]) -> Vector3<f32> {
    return Vector3::new(v[0], v[1], v[2]);
}

pub fn vec_from_array4(v : &[f32; 4]) -> Vector4<f32> {
    return Vector4::new(v[0], v[1], v[2], v[3]);
}

pub fn print_mat(m : &Matrix4<f32>) -> String {
    let mut res : String = "".to_string();
    res += &*format!("\n{:?}", m.row(0));
    res += &*format!("\n{:?}", m.row(1));
    res += &*format!("\n{:?}", m.row(2));
    res += &*format!("\n{:?}", m.row(3));
    return res;
}

pub fn print_mat_ui(m : &Matrix4<f32>, ui: &mut Ui) {
    ui.text(im_str!("{:+.3?}", m.row(0)));
    ui.text(im_str!("{:+.3?}", m.row(1)));
    ui.text(im_str!("{:+.3?}", m.row(2)));
    ui.text(im_str!("{:+.3?}", m.row(3)));
}
