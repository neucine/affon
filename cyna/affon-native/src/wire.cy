const maximum_frame_bytes: usize = 256 * 1024 * 1024;
const stdin_fd: c_int = 0;
const stdout_fd: c_int = 1;

extern c("read") fn os_read(fd: c_int, buffer: ?*c_void,
    count: c_size_t
): c_llong;
extern c("write") fn os_write(fd: c_int, buffer: ?*const c_void,
    count: c_size_t
): c_llong;

pub enum WireError: i32 {
    read_failed = 0,
    write_failed,
    invalid_count,
    no_progress,
    limit_exceeded,
}

fn os_read_some(output: []u8): c_llong ! WireError {
    var count: c_llong = -1;
    unchecked {
        count = os_read(stdin_fd, output.ptr as! *c_void,
            output.len() as c_size_t
        );
    }
    if count < 0 {
        throw WireError.read_failed;
    }
    return count;
}

fn os_write_all(input: []const u8): void ! WireError {
    var offset: usize = 0;
    while offset < input.len() {
        var count: c_llong = -1;
        unchecked {
            count = os_write(stdout_fd, input[offset..].ptr as! *const c_void,
                (input.len() - offset) as c_size_t
            );
        }
        if count <= 0 {
            throw WireError.write_failed;
        }
        const written = count as usize;
        if written > input.len() - offset {
            throw WireError.invalid_count;
        }
        offset = offset + written;
    }
}

fn read_exact(output: []u8): bool ! WireError {
    var offset: usize = 0;
    while offset < output.len() {
        const count = try os_read_some(output[offset..]);
        if count == 0 {
            return false;
        }
        const read = count as usize;
        if read > output.len() - offset {
            throw WireError.invalid_count;
        }
        offset = offset + read;
    }
    return true;
}

fn write_length(value: usize): void ! WireError {
    const header: [4]u8 = [
        (value & 0xff) as u8,
        ((value >> 8) & 0xff) as u8,
        ((value >> 16) & 0xff) as u8,
        ((value >> 24) & 0xff) as u8
    ];
    try os_write_all(header[..]);
}

fn read_length(header: []const u8): usize {
    return (header[0] as usize) |
    ((header[1] as usize) << 8) |
    ((header[2] as usize) << 16) |
    ((header[3] as usize) << 24);
}

pub fn read_frame(): ?List<u8> !WireError {
    var header: [4]u8 = [0, 0, 0, 0];
    if not try read_exact(header[..]) {
        return none;
    }
    const length = read_length(header[..]);
    if length > maximum_frame_bytes {
        throw WireError.limit_exceeded;
    }
    var frame: List<u8> = list<u8>();
    frame.reserve(length);
    var scratch: [8192]u8 = [];
    var remaining = length;
    while remaining != 0 {
        var requested: usize = 8192;
        if remaining < requested {
            requested = remaining;
        }
        const count = try os_read_some(scratch[..requested]);
        const read = count as usize;
        if read == 0 or read > requested {
            throw WireError.no_progress;
        }
        var index: usize = 0;
        while index < read {
            frame.push(scratch[index]);
            index = index + 1;
        }
        remaining = remaining - read;
    }
    return frame;
}

pub fn write_frame(metadata: string,
    payload: []const u8
): void ! WireError {
    const metadata_bytes = metadata.bytes();
    const length = 4 + metadata_bytes.len() + payload.len();
    if length > maximum_frame_bytes {
        throw WireError.limit_exceeded;
    }
    try write_length(length);
    try write_length(metadata_bytes.len());
    try os_write_all(metadata_bytes);
    try os_write_all(payload);
}

pub fn metadata_length(frame: []const u8): ?usize {
    if frame.len() < 4 {
        return none;
    }
    const length = read_length(frame[..4]);
    if length > frame.len() - 4 {
        return none;
    }
    return length;
}

pub struct JsonWriter {
    pub bytes: List<u8>;

    pub fn raw(self: &JsonWriter, text: string): void {
        for byte in text.bytes() {
            self.bytes.push(byte);
        }
    }

    pub fn string(self: &JsonWriter, text: []const u8): void {
        self.bytes.push(34);
        for byte in text {
            switch byte {
                34 => self.raw("\\\"");
                92 => self.raw("\\\\");
                8 => self.raw("\\b");
                12 => self.raw("\\f");
                10 => self.raw("\\n");
                13 => self.raw("\\r");
                9 => self.raw("\\t");
                _ => {
                    if byte < 32 {
                        self.raw("\\u00");
                        self.bytes.push(hex_digit((byte >> 4) as u8));
                        self.bytes.push(hex_digit(byte & 15));
                    } else {
                        self.bytes.push(byte);
                    }
                }
            }
        }
        self.bytes.push(34);
    }

    pub fn usize(self: &JsonWriter, value: usize): void {
        if value >= 10 {
            self.usize(value / 10);
        }
        self.bytes.push(48 + (value % 10) as u8);
    }

    pub fn integer(self: &JsonWriter, value: i64): void {
        if value < 0 {
            self.bytes.push(45);
            self.usize((0 - value) as usize);
        } else {
            self.usize(value as usize);
        }
    }

    pub fn boolean(self: &JsonWriter, value: bool): void {
        if value {
            self.raw("true");
        } else {
            self.raw("false");
        }
    }

    pub fn finish(self: JsonWriter): string {
        return string.from(self.bytes.view());
    }
}

fn hex_digit(value: u8): u8 {
    if value < 10 {
        return 48 + value;
    }
    return 87 + value;
}

pub fn json_writer(): JsonWriter {
    return JsonWriter {
        bytes: list<u8>()
    };
}
