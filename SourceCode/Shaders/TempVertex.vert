#version 330

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_color;
layout (location = 2) in vec2 vertex_texcoord;

out vec3 vs_position;
out vec3 vs_color;
out vec2 vs_texcoord;

//out TData
//{
//    vec3 vs_position;
//	vec3 vs_color;
//	vec2 vs_texcoord;
//	vec3 vs_normal;
//} outData;

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;
uniform mat4 ProjectionMatrix;

//void main()
//{
//	vs_position = vec4(ModelMatrix * vec4(vertex_position, 1.f)).xyz;                       //World space
//	vs_color    = vertex_color;
//	vs_texcoord = vec2(vertex_texcoord.x, vertex_texcoord.y * (-1.f));
//	vs_normal   = mat3(ModelMatrix) * vertex_normal;                                        //also in World space
//
//	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vertex_position, 1.f);  //MVP matrix
//	//Multiply SEQUENCE: from right to the left: Model Matrix at the first
//}				 

void main()
{
	vs_position = vec4(ModelMatrix * vec4(vertex_position, 1.f)).xyz;                       //World space
	vs_color    = vertex_color;
	vs_texcoord = vec2(vertex_texcoord.x, vertex_texcoord.y * (-1.f));
	gl_Position = ProjectionMatrix * ViewMatrix * ModelMatrix * vec4(vertex_position, 1.f);  //MVP matrix
	//Multiply SEQUENCE: from right to the left: Model Matrix at the first
}