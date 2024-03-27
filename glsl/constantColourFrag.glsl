#version 330 core

uniform vec4 colour;
out vec4 f_colour;

void main() {
	f_colour = colour;
}