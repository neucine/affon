import compute_core.lib as compute;
import hao.native_api as hao;

fn main(): i32 {
    const lhs: [2]f32 = [1.0, 2.0];
    const rhs: [2]f32 = [3.0, 4.0];
    const values = compute.add_f32(lhs[..], rhs[..]) catch (code) {
        return 3;
    };
    if values.len() != 2 or values[0] != 4.0 or values[1] != 6.0 {
        return 1;
    }
    if compute.api_version() != 1 or hao.api_version() != 1 {
        return 2;
    }
    print("compute=cyna hao=cyna sum=4,6\n");
    return 0;
}
