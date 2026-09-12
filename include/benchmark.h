#pragma once

#include <cstdint>


class Matrix;


class Benchmark {
    private:
        static constexpr float NO_COMPARISON = -1.0f;

        template<typename F>
        static void warmup(
            const Matrix &a,
            const Matrix &b,
            Matrix &c,
            F matMul,
            uint32_t numWarmups
        );

        template <typename F>
        static float runCpu(
            const Matrix &a,
            const Matrix &b,
            Matrix &c,
            F matMul,
            uint32_t numRuns
        );

        template <typename F>
        static float runCuda(
            const Matrix &a,
            const Matrix &b,
            Matrix &c,
            F matMul,
            uint32_t numRuns
        );

        static void printSep();
        static void printSpeedUp(float oldTime, float newTime);

    public:
        Benchmark() = delete;

        template <typename F>
        static float benchmarkMatMul(
            const Matrix &a,
            const Matrix &b,
            Matrix &c,
            const Matrix &gt,
            F matMul,
            uint32_t numRuns,
            uint32_t numWarmups,
            const char *testName,
            float compareTime = NO_COMPARISON
        );
};


#include "benchmark.tpp"
