#ifndef STDIO_H
#define STDIO_H

#include <stdint.h>

#define stderr 0
#define FILE void
void fprintf(FILE* f, const char *fmt, ...);

#endif