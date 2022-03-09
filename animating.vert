#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 mvp;
} ubo;

layout(set = 1, binding = 0) readonly buffer JointMatrices {
	mat4 jointMatrices[];
};

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout (location = 2) in vec4 inJointIndices;
layout (location = 3) in vec4 inJointWeights;

layout(location = 0) out vec2 fragTexCoord;

void main() {
	mat4 skinMat = 
		inJointWeights.x * jointMatrices[int(inJointIndices.x)] +
		inJointWeights.y * jointMatrices[int(inJointIndices.y)] +
		inJointWeights.z * jointMatrices[int(inJointIndices.z)] +
		inJointWeights.w * jointMatrices[int(inJointIndices.w)];
    gl_Position = ubo.mvp * skinMat * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
}
