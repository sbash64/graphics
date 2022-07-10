#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 mvp;
} ubo;

layout(push_constant) uniform PushConsts {
	mat4 model;
} primitive;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;

layout(location = 0) out vec2 fragTexCoord;

void main() {
    gl_Position = ubo.mvp * primitive.model * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
}
