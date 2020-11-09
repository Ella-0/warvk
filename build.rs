extern crate shaderc;

use std::env;
use std::io::Write;
use std::path::PathBuf;
fn main() {
    let vert_source = r#"
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColour;


const vec2 POSITIONS[4] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0)
);

layout(location = 0) out vec3 fragColour;

void main() {
    gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
    fragColour = inColour;
}
"#;

    let frag_source = r#"
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColour;

layout(location = 0) out vec4 outColour;

void main() {
    outColour = vec4(fragColour, 1.0);
    //outColour = vec4(1.0, 1.0, 1.0, 1.0);
}

"#;

    let mut compiler = shaderc::Compiler::new().unwrap();
    let vert_artifact = compiler
        .compile_into_spirv(
            vert_source,
            shaderc::ShaderKind::Vertex,
            "shader.glsl",
            "main",
            None,
        )
        .unwrap();

    let frag_artifact = compiler
        .compile_into_spirv(
            frag_source,
            shaderc::ShaderKind::Fragment,
            "shader.glsl",
            "main",
            None,
        )
        .unwrap();
    let vert_bytes = vert_artifact.as_binary_u8();
    let frag_bytes = frag_artifact.as_binary_u8();
    let vert_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("vert.spv");
    let frag_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("frag.spv");
    let mut vert_file = std::fs::File::create(vert_path).expect("create failed");
    let mut frag_file = std::fs::File::create(frag_path).expect("create failed");
    vert_file.write_all(vert_bytes).expect("write failed");
    frag_file.write_all(frag_bytes).expect("write failed");
}
