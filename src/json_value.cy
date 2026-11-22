import std.json;

pub type Value = json.Value;
pub type Object = json.Object;
pub type Array = json.Array;
pub type Number = json.Number;

pub fn field_index(object: &const Object, key: string): ?usize {
    return object.find(key);
}

pub fn string_field(object: &const Object, key: string): ?string {
    if field_index(object, key) is index {
        if object.members[index].value is .string(text) {
            return text;
        }
    }
    return none;
}

pub fn usize_field(object: &const Object, key: string): ?usize {
    if field_index(object, key) is index {
        if object.members[index].value is .number(number) {
            return parse_usize(number.spelling);
        }
    }
    return none;
}

pub fn bool_field(object: &const Object, key: string, fallback: bool): bool {
    if field_index(object, key) is index {
        if object.members[index].value is .bool(flag) {
            return flag;
        }
    }
    return fallback;
}

pub fn number_field(object: &const Object, key: string): ?f64 {
    if field_index(object, key) is index {
        if object.members[index].value is .number(number) {
            return parse_number(number.spelling);
        }
    }
    return none;
}

pub fn parse_usize(text: string): ?usize {
    var value: usize = 0;
    const bytes = text.bytes();
    if bytes.len() == 0 {
        return none;
    }
    var index: usize = 0;
    while index < bytes.len() {
        const byte = bytes[index];
        if byte < 48 or byte > 57 {
            return none;
        }
        const digit = (byte - 48) as usize;
        if value > ((0 as usize) -% 1 - digit) / 10 {
            return none;
        }
        value = value * 10 + digit;
        index = index + 1;
    }
    return value;
}

pub fn parse_number(text: string): ?f64 {
    const bytes = text.bytes();
    if bytes.len() == 0 {
        return none;
    }
    var negative = false;
    var index: usize = 0;
    if bytes[0] == 45 {
        negative = true;
        index = 1;
    }
    var value: f64 = 0.0;
    var digits: usize = 0;
    while index < bytes.len() {
        const byte = bytes[index];
        if byte == 46 or byte == 101 or byte == 69 {
            break;
        }
        if byte < 48 or byte > 57 {
            return none;
        }
        value = value * 10 + (byte - 48) as f64;
        digits = digits + 1;
        index = index + 1;
    }
    if digits == 0 {
        return none;
    }
    if index < bytes.len() and bytes[index] == 46 {
        index = index + 1;
        var place: f64 = 0.1;
        while index < bytes.len() and bytes[index] != 101 and bytes[index] != 69 {
            const byte = bytes[index];
            if byte < 48 or byte > 57 {
                return none;
            }
            value = value + (byte - 48) as f64 * place;
            place = place * 0.1;
            index = index + 1;
        }
    }
    var exponent: i32 = 0;
    var exponent_negative = false;
    if index < bytes.len() and (bytes[index] == 101 or bytes[index] == 69) {
        index = index + 1;
        if index < bytes.len() and (bytes[index] == 43 or bytes[index] == 45) {
            exponent_negative = bytes[index] == 45;
            index = index + 1;
        }
        if index >= bytes.len() {
            return none;
        }
        while index < bytes.len() {
            const byte = bytes[index];
            if byte < 48 or byte > 57 {
                return none;
            }
            exponent = exponent * 10 + (byte - 48) as i32;
            index = index + 1;
        }
    }
    while exponent > 0 {
        if exponent_negative {
            value = value * 0.1;
        } else {
            value = value * 10;
        }
        exponent = exponent - 1;
    }
    if negative {
        value = 0 - value;
    }
    return value;
}
