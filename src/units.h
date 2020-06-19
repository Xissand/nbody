#pragma once
#include <string>


constexpr int N = 512;
constexpr float cell_size = 4.0f; //ro=0.98

//constexpr int N = 2048;
//constexpr float cell_size = 6.349604208; //ro=0.98
//16.108111388
constexpr float v = 8 * cell_size * cell_size * cell_size;

constexpr float ro = N / v;

constexpr float T_INIT = 0.3;
float T_CONST = T_INIT;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL
#define EAM_POTENTIAL
//#define LJ_CUT

constexpr float ALPHA = 0.5;
constexpr float BETA = 6;
constexpr float Z0 = 12;

std::string off = "eam.xyz";

constexpr bool SNAP_XYZ = false;

//constexpr long long int total_steps = 5e5;
//constexpr int snap_steps = 100;
//constexpr int thermo_steps = 5000;
//constexpr float dt = 1e-4;

constexpr long long int total_steps = 30e5;
constexpr int snap_steps = 1000;
constexpr int thermo_steps = 0;
constexpr float dt = 1e-5;

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;

// ro = N/(2cell_size)**3