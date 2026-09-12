#include <cstdint>
#include <cstdio>
#include <string>
#include <iostream>

#include "benchmark.h"


constexpr uint32_t LINE_SIZE = 60;


void Benchmark::printSpeedUp(float oldTime, float newTime) {
    std::printf("%.2fx speed up\n", oldTime / newTime);
}


void Benchmark::printSep() {
    std::cout << std::string(LINE_SIZE, '=') << std::endl << std::endl;
}
