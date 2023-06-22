const std = @import("std");
const W = std.unicode.utf8ToUtf16LeStringLiteral;
const c = struct {
    usingnamespace @cImport({
        @cUndef("_DEBUG");
        @cInclude("Python.h");
        @cInclude("predict.h");
    });
};
extern fn MainEntry() void;

fn L(s: []const u8) [*:0]const u16 {
    var s_u16: [s.len + 1:0]u16 = undefined;
    for (0.., s) |i, v| {
        s_u16[i] = @as(u16, v);
    }
    s_u16[s.len] = 0;
    return &s_u16;
}


pub fn main() void {
    // const site_path = comptime L("./.venv/Lib/site-packages");
    const site_path = "./.venv/Lib/site-packages";
    if (c.PyImport_AppendInittab("predict", c.PyInit_predict) == -1) {
        @panic("unknow error");
    }
    c.Py_Initialize();
    defer c.Py_Finalize();
    const sys_path = c.PySys_GetObject("path");
    _ = c.PyList_Append(sys_path, c.PyUnicode_FromString(site_path));
    MainEntry();
}
