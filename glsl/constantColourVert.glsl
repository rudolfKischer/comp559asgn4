#version 330 core

uniform mat4 MVP;
uniform vec2 scale;
uniform vec2 offset;

in vec2 in_position;

void main() {	
	gl_Position = MVP * vec4( scale * in_position + offset, 0, 1.0);
}