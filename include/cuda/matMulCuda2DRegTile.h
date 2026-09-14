#pragma once


class Matrix;


void matMulCuda2DRegTileLaunch(const Matrix &a, const Matrix &b, Matrix &c);
