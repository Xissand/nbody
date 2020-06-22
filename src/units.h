#pragma once
#include <string>

//constexpr int N = 256;
//constexpr float cell_size = 4.0f; //ro=1 fcc

//constexpr int N = 1024;
//constexpr float cell_size = 5.0396842; //ro=1 bcc
//constexpr float cell_size = 4.489848193;

//constexpr int N = 512;
//constexpr float cell_size = 4.0f; //ro=1  DC

constexpr int N = 511;
constexpr float cell_size = 5.0f;

//constexpr int N = 2048;
//constexpr float cell_size = 6.349604208; //ro=1 fcc
////16.108111388
constexpr float v = 8 * cell_size * cell_size * cell_size;

constexpr float ro = N / v;

constexpr float T_INIT = 0.0;
float T_CONST = T_INIT;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL
#define EAM_POTENTIAL
//#define LJ_CUT

//constexpr float ALPHA = 10;//1.4; //A=10, B=7, T=0.001: stable dc
//constexpr float BETA = 7;

constexpr float ALPHA = 1.5;//1.4; //A=10, B=7, T=0.001: stable dc
constexpr float BETA = 6;
constexpr float Z0 = 12;

std::string off = "research/eam/melting2/1.xyz";

constexpr bool SNAP_XYZ = true;

//constexpr long long int total_steps = 5e5;
//constexpr int snap_steps = 100;
//constexpr int thermo_steps = 5000;
//constexpr float dt = 1e-4;

constexpr long long int total_steps = 2;
constexpr int snap_steps = 1;
constexpr int thermo_steps = 1e6;
constexpr float dt = 1e-3;

constexpr int BLOCK_SIZE = 511;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;

// ro = N/(2cell_size)**3