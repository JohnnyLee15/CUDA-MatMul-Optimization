#include <cstdint>
#include <cstdio>
#include <string>
#include <iostream>

#include "benchmark.h"


void Benchmark::printSpeedUp(float oldTime, float newTime, const char *speedUpOver) {
    std::printf("%.2fx speed up over %s\n", oldTime / newTime, speedUpOver);
}


void Benchmark::printSep(char sepChar, bool newline) {
    std::cout << std::string(LINE_SIZE, sepChar) << std::endl;
    if (newline) std::cout << std::endl;
}
