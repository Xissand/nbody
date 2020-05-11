#pragma once
#include <string>

constexpr int N = 4096;
constexpr float cell_size = 8.054055694; //ro=0.98
//16.108111388
constexpr float v = 8 * cell_size * cell_size * cell_size;

constexpr float ro = N / v;

constexpr float T_INIT = 2.0;
float T_CONST = T_INIT;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL
//#define LJ_CUT

std::string off = "glass.xyz";

constexpr bool SNAP_XYZ = true;

//constexpr long long int total_steps = 5e5;
//constexpr int snap_steps = 100;
//constexpr int thermo_steps = 5000;
//constexpr float dt = 1e-4;

constexpr long long int total_steps = 1e5;
constexpr int snap_steps = 10;
constexpr int thermo_steps = 1000;
constexpr float dt = 1e-4;

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;

// ro = N/(2cell_size)**3