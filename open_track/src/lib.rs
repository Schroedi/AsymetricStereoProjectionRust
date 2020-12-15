use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use std::{mem, thread};

pub struct OpenTrackServer {
    pos_rot: RwLock<Option<([f32; 3], [f32; 3])>>,
}

impl OpenTrackServer {
    pub fn start(port: Option<u16>) -> Arc<OpenTrackServer> {
        let addr = SocketAddr::from(([192, 168, 178, 24], port.unwrap_or(4242)));
        let socket = UdpSocket::bind(addr).expect("OpenTrack: Failed to bind socket");
        socket
            .set_read_timeout(Option::from(Duration::from_millis(10)))
            .expect("OpenTrack: Failed to set socket time-out");

        let server = Arc::new(OpenTrackServer {
            pos_rot: RwLock::new(Option::None),
        });

        let threads_server_ref = server.clone();
        thread::spawn(move || loop {
            threads_server_ref.update(&socket);
        });

        server.clone()
    }

    pub fn get_pos_rot(&self) -> Option<([f32; 3], [f32; 3])> {
        *self.pos_rot.read().unwrap()
    }

    fn update(&self, socket: &UdpSocket) {
        // 6 doubles per package
        let mut buf = [0; 6 * mem::size_of::<f64>()];
        match socket.recv(&mut buf) {
            Ok(n) => {
                assert_eq!(n, buf.len());
                let pos_rot = parse_pos_rot(&buf);
                let mut write = self.pos_rot.write().unwrap();
                *write = Option::from(pos_rot);
            }
            Err(e) => {
                let mut write = self.pos_rot.write().unwrap();
                *write = Option::None;
                eprintln!("Opentrack error receiving package: {}", e)
            },
        }
    }
}

fn parse_pos_rot(data: &[u8]) -> ([f32; 3], [f32; 3]) {
    (
        [
            read_f64_as_f32(data, 0),
            read_f64_as_f32(data, 1),
            read_f64_as_f32(data, 2),
        ],
        [
            read_f64_as_f32(data, 3),
            read_f64_as_f32(data, 4),
            read_f64_as_f32(data, 5),
        ],
    )
}

fn read_f64_as_f32(d: &[u8], idx: isize) -> f32 {
    unsafe { (d.as_ptr() as *const f64).offset(idx).read_unaligned() as f32 }
}

#[cfg(test)]
mod tests {
    use crate::OpenTrackServer;

    #[test]
    fn it_works() {
        let tracker = OpenTrackServer::start(None);
        let (pos, rot) = tracker.get_pos_rot();
        println!("pos: {:?}, rot: {:?}", pos, rot);
    }
}
