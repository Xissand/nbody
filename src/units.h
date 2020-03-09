#pragma once

constexpr int N = 512;
//constexpr float cell_size = 4;

constexpr float cell_size = 4.742524406; //ro= 0.6

constexpr float v = 8 * cell_size * cell_size * cell_size;


constexpr float ro = N / v;

constexpr float T_INIT = 2;
constexpr float T_CONST = 2;

#define PERIODIC_BOUNDARIES
#define LENNARD_JONES_POTENTIAL
#define LJ_CUT

constexpr bool SNAP_XYZ = false;

constexpr long long int total_steps = 8e5;
constexpr int snap_steps = 50;
constexpr int thermo_steps = 10;
constexpr float dt = 1e-6;

constexpr int BLOCK_SIZE = 128;

constexpr float NA = 1;
constexpr float K_B = 1;
constexpr float EPSILON0 = 1;
constexpr float MU0 = 1;

// ro = 0.6
// constexpr int N = 200;
// constexpr float cell_size = 3.466806372;

// constexpr int N = 250;
// constexpr float cell_size = 3.734503955;

// constexpr int N = 300;
// constexpr float cell_size = 3.96850263;

// constexpr int N = 350;
// constexpr float cell_size = 4.177748279;

// constexpr int N = 400;
// constexpr float cell_size = 4.367902324;

// constexpr int N = 500;
// constexpr float cell_size = 4.705180144;

// constexpr int N = 600;
// constexpr float cell_size = 5.0f;

// constexpr int N = 700;
// constexpr float cell_size = 5.263632998;

// ro = N/(2cell_size)**3

// constexpr int N = 512;
// constexpr float cell_size = 4.142976675;// ro = 0.9
// constexpr float cell_size = 4.30886938;// ro = 0.8
// constexpr float cell_size = 4.504991522;// ro = 0.7
// constexpr float cell_size = 4.742524406; //to= 0.6
// constexpr float cell_size = 5.0396842;// ro = 0.5
// constexpr float cell_size = 5.428835233;// ro = 0.4
// constexpr float cell_size = 5.975206329;// ro = 0.3
// constexpr float cell_size = 6.839903787;// ro = 0.2
// constexpr float cell_size = 8.61773876;// ro = 0.1
