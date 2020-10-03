use std::fs::File;
use std::io;
use std::io::Read;
use std::mem;

use std::os::unix::io::AsRawFd;

use crate::ctx::Ctx;
use smithay::reexports::wayland_server::calloop::LoopHandle;

#[derive(Debug)]
#[repr(C)]
pub struct InputEvent {
    tv_sec: isize,  // from timeval struct
    tv_usec: isize, // from timeval struct
    pub type_: u16,
    pub code: u16,
    pub value: i32,
}

const MAX_KEYS: u16 = 112;

const UK: &'static str = "<UK>";

const KEY_NAMES: [&'static str; MAX_KEYS as usize] = [
    UK,
    "<ESC>",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "0",
    "-",
    "=",
    "<Backspace>",
    "<Tab>",
    "q",
    "w",
    "e",
    "r",
    "t",
    "y",
    "u",
    "i",
    "o",
    "p",
    "[",
    "]",
    "<Enter>",
    "<LCtrl>",
    "a",
    "s",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    ";",
    "'",
    "`",
    "<LShift>",
    "\\",
    "z",
    "x",
    "c",
    "v",
    "b",
    "n",
    "m",
    ",",
    ".",
    "/",
    "<RShift>",
    "<KP*>",
    "<LAlt>",
    " ",
    "<CapsLock>",
    "<F1>",
    "<F2>",
    "<F3>",
    "<F4>",
    "<F5>",
    "<F6>",
    "<F7>",
    "<F8>",
    "<F9>",
    "<F10>",
    "<NumLock>",
    "<ScrollLock>",
    "<KP7>",
    "<KP8>",
    "<KP9>",
    "<KP->",
    "<KP4>",
    "<KP5>",
    "<KP6>",
    "<KP+>",
    "<KP1>",
    "<KP2>",
    "<KP3>",
    "<KP0>",
    "<KP.>",
    UK,
    UK,
    UK,
    "<F11>",
    "<F12>",
    UK,
    UK,
    UK,
    UK,
    UK,
    UK,
    UK,
    "<KPEnter>",
    "<RCtrl>",
    "<KP/>",
    "<SysRq>",
    "<RAlt>",
    UK,
    "<Home>",
    "<Up>",
    "<PageUp>",
    "<Left>",
    "<Right>",
    "<End>",
    "<Down>",
    "<PageDown>",
    "<Insert>",
    "<Delete>",
];

const SHIFT_KEY_NAMES: [&'static str; MAX_KEYS as usize] = [
    UK,
    "<ESC>",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "_",
    "+",
    "<Backspace>",
    "<Tab>",
    "Q",
    "W",
    "E",
    "R",
    "T",
    "Y",
    "U",
    "I",
    "O",
    "P",
    "{",
    "}",
    "<Enter>",
    "<LCtrl>",
    "A",
    "S",
    "D",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    ":",
    "\"",
    "~",
    "<LShift>",
    "|",
    "Z",
    "X",
    "C",
    "V",
    "B",
    "N",
    "M",
    "<",
    ">",
    "?",
    "<RShift>",
    "<KP*>",
    "<LAlt>",
    " ",
    "<CapsLock>",
    "<F1>",
    "<F2>",
    "<F3>",
    "<F4>",
    "<F5>",
    "<F6>",
    "<F7>",
    "<F8>",
    "<F9>",
    "<F10>",
    "<NumLock>",
    "<ScrollLock>",
    "<KP7>",
    "<KP8>",
    "<KP9>",
    "<KP->",
    "<KP4>",
    "<KP5>",
    "<KP6>",
    "<KP+>",
    "<KP1>",
    "<KP2>",
    "<KP3>",
    "<KP0>",
    "<KP.>",
    UK,
    UK,
    UK,
    "<F11>",
    "<F12>",
    UK,
    UK,
    UK,
    UK,
    UK,
    UK,
    UK,
    "<KPEnter>",
    "<RCtrl>",
    "<KP/>",
    "<SysRq>",
    "<RAlt>",
    UK,
    "<Home>",
    "<Up>",
    "<PageUp>",
    "<Left>",
    "<Right>",
    "<End>",
    "<Down>",
    "<PageDown>",
    "<Insert>",
    "<Delete>",
];

pub fn get_key_text(code: u16, shift_pressed: u8) -> &'static str {
    let arr = if shift_pressed != 0 {
        SHIFT_KEY_NAMES
    } else {
        KEY_NAMES
    };

    if code < MAX_KEYS {
        return arr[code as usize];
    } else {
        //println!("Unknown key: {}", code);
        return UK;
    }
}

const EV_KEY: u16 = 1;

pub fn is_key_event(type_: u16) -> bool {
    type_ == EV_KEY
}

const KEY_PRESS: i32 = 1;

pub fn is_key_press(value: i32) -> bool {
    value == KEY_PRESS
}

pub fn init<W>(loop_handle: LoopHandle<Ctx<W>>) -> std::sync::mpsc::Receiver<InputEvent>
where
    W: Send + Sync + 'static,
{
    println!("╠══ kbd init");

    let (tx, rx) = std::sync::mpsc::channel::<InputEvent>();

    let devnode = "/dev/input/by-path/platform-i8042-serio-0-event-kbd";

    let mut file = File::open(devnode).expect("Couldn't Open Keyboard");
    let mut buf: [u8; mem::size_of::<InputEvent>()] = unsafe { mem::zeroed() };
    //    let mut buf_reader = io::BufReader::new(file);

    /*    std::thread::spawn(move || loop {
        let count = buf_reader.read_exact(&mut buf).expect("Read Failed");
        let event: InputEvent = unsafe { mem::transmute(buf) };
        tx.send(event);
    });*/

/*    loop_handle.insert_source(
        calloop::generic::Generic::from_fd(
            file.as_raw_fd(),
            calloop::Interest::Readable,
            calloop::Mode::Level,
        ),
        {
            move |_, _, ctx: &mut Ctx<W>| {
                println!("called");
                Ok(())
            }
        },
    );*/

    return rx;
}
