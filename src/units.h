#pragma once

//ro = N/(2cell_size)**3

constexpr int N = 512;

constexpr float cell_size = 4.142976675;// ro = 0.9
//constexpr float cell_size = 4.30886938;// ro = 0.8
//constexpr float cell_size = 4.504991522;// ro = 0.7
//constexpr float cell_size = 4.742524406; //to= 0.6
//constexpr float cell_size = 5.0396842;// ro = 0.5
//constexpr float cell_size = 5.428835233;// ro = 0.4
//constexpr float cell_size = 5.975206329;// ro = 0.3
//constexpr float cell_size = 6.839903787;// ro = 0.2
//constexpr float cell_size = 8.61773876;// ro = 0.1

constexpr float T_INIT = 2;
constexpr float T_CONST = 2;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL

constexpr bool SNAP_XYZ = false;

constexpr long long int total_steps = 1e6;
constexpr int snap_steps = 50;
constexpr int thermo_steps = 10;
constexpr float dt = 1e-6; 

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;
