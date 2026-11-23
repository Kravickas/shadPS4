// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Shared by the camera solve shaders. The including shader declares `double sums[]` in the
// layout written by xfb_camera_sums.comp.

const uint RegionStride = 32u;
const uint NormalCount = 26u;
// Static geometry measures near 1e-10; moving and skinned geometry is orders of magnitude above.
const double InlierResidual = 1e-5LF;
const double PivotEpsilon = 1e-12LF;
const uint UpperStart[4] = uint[](0u, 4u, 7u, 9u);

double Sum(uint region, uint slot) {
    return sums[region * RegionStride + slot];
}

double Trace(uint region) {
    return Sum(region, 0u) + Sum(region, 4u) + Sum(region, 7u) + Sum(region, 9u);
}

bool Usable(uint region) {
    return Sum(region, 27u) > 0.0LF && Trace(region) > 0.0LF;
}

double Upper(uint region, uint i, uint j) {
    const uint lo = min(i, j);
    const uint hi = max(i, j);
    return Sum(region, UpperStart[lo] + (hi - lo));
}

// sum |p - C c|^2 relative to sum |c|^2.
double RelativeResidual(uint region, dvec4 c[4]) {
    double r = Sum(region, 26u);
    for (uint row = 0u; row < 4u; ++row) {
        for (uint i = 0u; i < 4u; ++i) {
            r -= 2.0LF * c[row][i] * Sum(region, 10u + row * 4u + i);
            for (uint j = 0u; j < 4u; ++j) {
                r += c[row][i] * Upper(region, i, j) * c[row][j];
            }
        }
    }
    return max(r, 0.0LF) / Trace(region);
}

// Unit trace per region, so dense meshes do not outweigh large surfaces.
void Accumulate(uint region, inout double normal[NormalCount]) {
    const double inv_trace = 1.0LF / Trace(region);
    for (uint k = 0u; k < NormalCount; ++k) {
        normal[k] += Sum(region, k) * inv_trace;
    }
}

// Gauss-Jordan with partial pivoting on [A | B^T]; column r of the solution is row r of C.
bool SolveNormal(double normal[NormalCount], out dvec4 c[4]) {
    double m[4][8];
    double scale = 0.0LF;
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 4u; ++j) {
            const uint lo = min(i, j);
            const uint hi = max(i, j);
            m[i][j] = normal[UpperStart[lo] + (hi - lo)];
        }
        for (uint r = 0u; r < 4u; ++r) {
            m[i][4u + r] = normal[10u + r * 4u + i];
        }
        scale = max(scale, abs(m[i][i]));
    }
    for (uint r = 0u; r < 4u; ++r) {
        c[r] = dvec4(0.0LF);
        c[r][r] = 1.0LF;
    }
    if (scale <= 0.0LF) {
        return false;
    }
    for (uint col = 0u; col < 4u; ++col) {
        uint pivot = col;
        double best = abs(m[col][col]);
        for (uint row = col + 1u; row < 4u; ++row) {
            if (abs(m[row][col]) > best) {
                best = abs(m[row][col]);
                pivot = row;
            }
        }
        if (best <= PivotEpsilon * scale) {
            return false;
        }
        for (uint k = 0u; k < 8u; ++k) {
            const double t = m[col][k];
            m[col][k] = m[pivot][k];
            m[pivot][k] = t;
        }
        for (uint row = 0u; row < 4u; ++row) {
            if (row != col) {
                const double f = m[row][col] / m[col][col];
                for (uint k = 0u; k < 8u; ++k) {
                    m[row][k] -= f * m[col][k];
                }
            }
        }
    }
    for (uint r = 0u; r < 4u; ++r) {
        c[r] = dvec4(m[0][4u + r] / m[0][0], m[1][4u + r] / m[1][1], m[2][4u + r] / m[2][2],
                     m[3][4u + r] / m[3][3]);
    }
    return true;
}
