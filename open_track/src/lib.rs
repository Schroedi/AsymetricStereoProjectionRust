use std::{mem, thread};
use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, RwLock};
use std::time::Duration;

pub struct OpenTrackServer {
    buf: RwLock<[u8; 6 * mem::size_of::<f64>()]>,
}

impl OpenTrackServer {
    pub fn start(port: Option<u16>) -> Arc<OpenTrackServer> {
        let server = Arc::new(OpenTrackServer {
            buf: RwLock::new([0; 6 * mem::size_of::<f64>()]),
        });

        // UDP socket
        let addr = SocketAddr::from(([192, 168, 178, 24], port.unwrap_or(4242)));
        let socket = UdpSocket::bind(addr).
            expect("OpenTrack: Failed to bind socket");
        socket.set_read_timeout(Option::from(Duration::from_millis(10))).
            expect("OpenTrack: Failed to set socket time-out");

        // update position in background thread
        {
            let threads_server_ref = server.clone();
            thread::spawn(move || loop {
                threads_server_ref.update(&socket);
            });
        }

        server.clone()
    }

    pub fn get_pos_rot(&self) -> ([f32; 3], [f32; 3]) {
        let buf = *self.buf.read().unwrap();
        ([
             read_f64_as_f32(&buf, 0),
             read_f64_as_f32(&buf, 1),
             read_f64_as_f32(&buf, 2),
         ], [
             read_f64_as_f32(&buf, 3),
             read_f64_as_f32(&buf, 4),
             read_f64_as_f32(&buf, 5),
         ])
    }

    fn update(&self, socket: &UdpSocket) {
        // 6 doubles per package
        let mut buf = [0; 6 * mem::size_of::<f64>()];
        match socket.recv(&mut buf) {
            Ok(n) => {
                assert_eq!(n, buf.len());
                {
                    let mut target = self.buf.write().unwrap();
                    target.copy_from_slice(&buf);
                }
            }
            Err(e) => eprintln!("Opentrack error receiving package: {}", e),
        }
    }
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
