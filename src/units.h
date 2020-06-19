#pragma once
#include <string>

//constexpr int N = 256;
//constexpr float cell_size = 3.174802104; //ro=1

//constexpr int N = 1024;
//constexpr float cell_size = 5.0396842; //ro=1

constexpr int N = 512;
constexpr float cell_size = 4.0f; //ro=1

//constexpr int N = 2048;
//constexpr float cell_size = 6.349604208; //ro=1
//16.108111388
constexpr float v = 8 * cell_size * cell_size * cell_size;

constexpr float ro = N / v;

constexpr float T_INIT = 0.1;
float T_CONST = T_INIT;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL
#define EAM_POTENTIAL
//#define LJ_CUT

constexpr float ALPHA = 1.4;
constexpr float BETA = 7;
constexpr float Z0 = 12;


std::string off = "eam.xyz";

constexpr bool SNAP_XYZ = true;

//constexpr long long int total_steps = 5e5;
//constexpr int snap_steps = 100;
//constexpr int thermo_steps = 5000;
//constexpr float dt = 1e-4;

constexpr long long int total_steps = 1e5;
constexpr int snap_steps = 1000;
constexpr int thermo_steps = 10;
constexpr float dt = 1e-5;

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;

// ro = N/(2cell_size)**3