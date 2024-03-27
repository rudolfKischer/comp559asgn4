#version 330 core

uniform mat4 MVP;

in vec2 in_position;
in vec4 in_colour;

out vec4 colour;

void main() {	
	gl_Position = MVP * vec4(in_position, 0, 1.0);
	colour = in_colour;
}