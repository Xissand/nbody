#pragma once

constexpr int N = 4096;
constexpr int cell_size = 8;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL

constexpr int total_steps = 1e6;
constexpr int snap_steps = 5;
constexpr float dt = 1e-6; 

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float k = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;
