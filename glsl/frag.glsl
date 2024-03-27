#version 330 core

out vec4 f_color;
in vec4 colour;

void main() {
	f_color = colour;
}