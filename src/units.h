#pragma once

constexpr int N = 8;
constexpr float cell_size = 1;

constexpr float T_INIT = 100;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL

constexpr bool SNAP_XYZ = true;

constexpr int total_steps = 3e9;
constexpr int snap_steps = 500;
constexpr float dt = 1e-15; 

constexpr int BLOCK_SIZE = 8;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;
