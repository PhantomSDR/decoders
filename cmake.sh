#!/usr/bin/env bash
if [[ $* == *--build* ]]
then
cmake "$@"
else
cmake -DCMAKE_C_BYTE_ORDER=LITTLE_ENDIAN "$@"
fi

