#pragma once

constexpr int N = 512;
constexpr float cell_size = 4;

constexpr float T_INIT = 100;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL

constexpr bool SNAP_XYZ = false;

constexpr long long int total_steps = 1e5;
constexpr int snap_steps = 500;
constexpr float dt = 1e-6; 

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;
