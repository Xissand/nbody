#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "molecules.h"
#include "units.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

int step = 0;

float4* device_q;
float4* device_v;
float4* host_q;
float4* host_v;
float4* device_e;
float4* host_e;
Molecule* device_mol;
Molecule* host_mol;
float4 params;

__device__ void get_force(float4 qi, float4 qj, Molecule moli, float3& fi)
{
    float3 r;
    float coeff = 1;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

#ifdef PERIODIC_BOUNDARIES
    r.x -= roundf(r.x / (2 * cell_size)) * (2 * cell_size);
    r.y -= roundf(r.y / (2 * cell_size)) * (2 * cell_size);
    r.z -= roundf(r.z / (2 * cell_size)) * (2 * cell_size);
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;

#ifdef LJ_CUT // Cutoff set to half the box length
    if (r2 > cell_size * cell_size)
        return;
#endif

    float k = 0;

#ifdef LENNARD_JONES_POTENTIAL
    r2 = r2 / (moli.SIGMA * moli.SIGMA);
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r4 * r4;

    k += 4 * moli.EPSILON * (-(0.5f / r8) + (1.0f / (r6 * r8))) * 12;
#endif

#ifdef EAM_POTENTIAL
    float r_mod = sqrtf(r2) * powf(2, -1/6);

    float df = BETA * expf(-BETA * (r_mod - 1));
    float dens = qi.w * qj.w / (Z0 * Z0);///qj.w;

    k += powf(2, -1/6)* ALPHA * 0.5 * df * logf(dens) / r_mod;

    k += powf(2, -1/6)* ALPHA * BETA * (r_mod - 1) * df / r_mod;

#endif

#ifdef COULOMB_POTENTIAL
    if (r2 < 0.0001)
        r2 = 0.0001;

    float r_mod = sqrtf(r2);
    float r3 = r_mod * r_mod * r_mod;

    k += 1.0f / r3;
#endif

    fi.x += r.x * k;
    fi.y += r.y * k;
    fi.z += r.z * k;
}

__device__ void get_densities(float4 qi, float4 qj, float& dens)
{
    float3 r;
    float coeff = 1;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

#ifdef PERIODIC_BOUNDARIES
    r.x -= roundf(r.x / (2 * cell_size)) * (2 * cell_size);
    r.y -= roundf(r.y / (2 * cell_size)) * (2 * cell_size);
    r.z -= roundf(r.z / (2 * cell_size)) * (2 * cell_size);
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    float r_mod = sqrtf(r2) * powf(2, -1/6);

#ifdef LJ_CUT // Cutoff set to half the box length
    if (r2 > (cell_size * cell_size))
    {
        return;
    }
#endif

    coeff = -BETA * (r_mod - 1);
    float k = expf(coeff);

    // printf("%f\n", r_mod);

    dens += k;
}

// TODO: IMPLEMENT EAM
__device__ void get_virial(float4 qi, float4 qj, Molecule moli, float4& e)
{
    float3 r;
    float coeff = 1;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

#ifdef PERIODIC_BOUNDARIES
    r.x -= roundf(r.x / (2 * cell_size)) * (2 * cell_size);
    r.y -= roundf(r.y / (2 * cell_size)) * (2 * cell_size);
    r.z -= roundf(r.z / (2 * cell_size)) * (2 * cell_size);
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;

#ifdef LJ_CUT // Cutoff set to half the box length
    if (r2 > (cell_size * cell_size))
    {
        return;
    }
#endif
float k = 0;
#ifdef LENNARD_JONES_POTENTIAL
    r2 = r2 / (moli.SIGMA * moli.SIGMA);
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r4 * r4;

    coeff = 4 * moli.SIGMA;
     k = (-(0.5f) + (1.0f / r6)) * 12 / r8;
#endif

#ifdef COULOMB_POTENTIAL
    if (r2 < 0.0001)
        r2 = 0.0001;

    float r_mod = sqrtf(r2);
    float r3 = r_mod * r_mod * r_mod;

    k += 1.0f / r3;
#endif

    float3 temp;
    temp.x = r.x * coeff * k;
    temp.y = r.y * coeff * k;
    temp.z = r.z * coeff * k;

    k = temp.x * r.x + temp.y * r.y + temp.z * r.z;
    e.w += k;
}

__device__ void get_potential(float4 qi, float4 qj, Molecule moli, float4& e)
{
    float p = 0;
    float3 r;
    float coeff = 1;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

#ifdef PERIODIC_BOUNDARIES
    r.x -= roundf(r.x / (2 * cell_size)) * (2 * cell_size);
    r.y -= roundf(r.y / (2 * cell_size)) * (2 * cell_size);
    r.z -= roundf(r.z / (2 * cell_size)) * (2 * cell_size);
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;

#ifdef LJ_CUT // Cutoff set to half the box length
    if (r2 > cell_size * cell_size)
        return;
#endif

#ifdef LENNARD_JONES_POTENTIAL
    r2 = r2 / (moli.SIGMA * moli.SIGMA);
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    coeff = 4 * moli.EPSILON;

    p = (-(1.0f) + (1.0f / r6)) * coeff / r6;

    e.x += p / 2.0f;
    e.z += p / 2.0f;
#endif

#ifdef EAM_POTENTIAL
    float r_mod = sqrtf(r2)  * powf(2, -1/6);
    // float f_comp = ALPHA * 0.5 * qi.w * (logf(qi.w) - logf(Z0) - 1);
    float ex = -BETA * (r_mod - 1);
    float f_pair = -ALPHA * 0.5 * expf(ex) * (ex - 1);

    // printf("%f %f\n",f_comp, f_pair);

    e.x += f_pair; // + f_comp / (N - 1);
    e.z += f_pair; // + f_comp / (N - 1);

#endif

#ifdef COULOMB_POTENTIAL
    if (r2 < 0.0001)
        r2 = 0.0001;

    float r_mod = sqrtf(r2);

    p += 1.0f / r_mod;
    e.x += p / 2;
    e.z += p / 2;
#endif
}

__device__ void get_kinetic(float4 v, Molecule mol, float4& e)
{
    float k = (v.x * v.x + v.y * v.y + v.z * v.z) / 2.0f;
    e.y += mol.M * k;
    e.z += mol.M * k;
}

__global__ void evolve_densities(float4* d_q, int N_BODIES)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float4 shared_q[BLOCK_SIZE];
    float4 q = d_q[j];
    float dens = 0;

    for (int i = 0; i < N_BODIES; i += BLOCK_SIZE)
    {
        shared_q[threadIdx.x] = d_q[i + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            if (k + i == j)
                continue;

            get_densities(q, shared_q[k], dens);
        }
    }

    q.w = dens;
    __syncthreads();

    d_q[j] = q;

    // printf("%f\n",q.w);
}

__global__ void evolve(float4* d_q, float4* d_v, Molecule* d_mol, int N_BODIES, float dt, float4* d_e, bool snap)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float4 shared_q[BLOCK_SIZE];
    float4 q = d_q[j];
    float4 v = d_v[j];
    Molecule mol = d_mol[j];
    float4 e = {0.0f, 0.0f, 0.0f, 0.0f};
    float3 f = {0.0f, 0.0f, 0.0f};
    float3 a = {0.0f, 0.0f, 0.0f};
    float e_pot = 0;

    for (int i = 0; i < N_BODIES; i += BLOCK_SIZE)
    {
        shared_q[threadIdx.x] = d_q[i + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            if (k + i == j)
                continue;

            get_force(q, shared_q[k], mol, f);
            if (snap)
            {
                get_potential(q, shared_q[k], mol, e);
                get_virial(q, shared_q[k], mol, e);
            }
        }
        __syncthreads();
    }
    if (snap)
        get_kinetic(v, mol, e);

    float m = mol.M;
    a.x = f.x / m;
    a.y = f.y / m;
    a.z = f.z / m;

    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;

    __syncthreads();

#ifdef EAM_POTENTIAL
    float f_comp = ALPHA * 0.5 * q.w * (logf(q.w) - logf(Z0) - 1);

    e.x += f_comp;
    e.z += f_comp;
#endif

    q.x += v.x * dt;
    q.y += v.y * dt;
    q.z += v.z * dt;
    q.w = 0;

// Reflective boundaries
#ifdef REFLECTIVE_BOUNDARIES
    if (q.x > cell_size)
    {
        q.x -= 2 * (q.x - cell_size);
        v.x = -v.x;
    }
    if (q.x < -cell_size)
    {
        q.x += 2 * (-q.x - cell_size);
        v.x = -v.x;
    }
    if (q.y > cell_size)
    {
        q.y -= 2 * (q.y - cell_size);
        v.y = -v.y;
    }
    if (q.y < -cell_size)
    {
        q.y += 2 * (-q.y - cell_size);
        v.y = -v.y;
    }
    if (q.z > cell_size)
    {
        q.z -= 2 * (q.z - cell_size);
        v.z = -v.z;
    }
    if (q.z < -cell_size)
    {
        q.z += 2 * (-q.z - cell_size);
        v.z = -v.z;
    }
#endif

    __syncthreads();

    d_q[j] = q;
    d_v[j] = v;
    d_e[j] = e;
}

void thermostat_scale()
{
    float E_KIN0 = 0;
    cudaMemcpy(host_v, device_v, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_e, device_e, sizeof(float4) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        E_KIN0 += host_e[i].y;

    float T = E_KIN0 * 2 / (3 * N * K_B);

    //T_CONST = T_INIT * (1.0 - ((float) step / total_steps));

    // cout << T_CONST << endl;

    float coeff = sqrt(T_CONST / T);

    for (int i = 0; i < N; i++)
    {
        host_v[i].x *= coeff;
        host_v[i].y *= coeff;
        host_v[i].z *= coeff;
    }
    cudaMemcpy(device_v, host_v, sizeof(float4) * N, cudaMemcpyHostToDevice);
}

void generate()
{
    int limit = (int) truncf(cbrtf(N)) + 1;
    float grid_size = (2 * cell_size / limit);

    bool grid[limit][limit][limit];
    for (int i = 0; i < limit; i++)
        for (int j = 0; j < limit; j++)
            for (int k = 0; k < limit; k++)
                grid[i][j][k] = false;

    for (int n = 0; n < N; n++)
    {
        int grid_x = rand() % limit;
        int grid_y = rand() % limit;
        int grid_z = rand() % limit;

        if (grid[grid_x][grid_y][grid_z])
        {
            n--;
            continue;
        }

        float alpha = 0.000; // How much on grid particle are

        host_q[n].x = grid_x * grid_size + ((float) rand() / RAND_MAX) * alpha * grid_size - cell_size;
        host_q[n].y = grid_y * grid_size + ((float) rand() / RAND_MAX) * alpha * grid_size - cell_size;
        host_q[n].z = grid_z * grid_size + ((float) rand() / RAND_MAX) * alpha * grid_size - cell_size;
        host_q[n].w = 0;

        host_mol[n].set((MOLECULES) DEFAULT);

        // Rms for Maxwell disftibution is (3kT/m)**0.5, for rand is 1
        float velocity_coeff = sqrt((3 * K_B * T_INIT) / host_mol[n].M);

        host_v[n].x = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
        host_v[n].y = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
        host_v[n].z = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
        host_v[n].w = 1;

        // TODO: calculate starting energy
        host_e[n].x = 0; // Potential
        host_e[n].y = 0; // Kinetic
        host_e[n].z = 0; // Total
        host_e[n].w = 0; // Virial

        grid[grid_x][grid_y][grid_z] = true;
    }
}

void generate_fcc()
{
    int limit = 4; //(int) (cbrtf(N / 4));
    float grid_size = (2 * cell_size / limit);

    for (int grid_x = 0; grid_x < limit; grid_x++)
        for (int grid_y = 0; grid_y < limit; grid_y++)
            for (int grid_z = 0; grid_z < limit; grid_z++)
            {
                int n = grid_x * limit * limit + grid_y * limit + grid_z;

                host_q[n * 4].x = grid_x * grid_size;
                host_q[n * 4].y = grid_y * grid_size;
                host_q[n * 4].z = grid_z * grid_size;
                host_q[n * 4].w = 0;

                host_q[n * 4 + 1].x = grid_x * grid_size + 0.5 * grid_size;
                host_q[n * 4 + 1].y = grid_y * grid_size + 0.5 * grid_size;
                host_q[n * 4 + 1].z = grid_z * grid_size;
                host_q[n * 4 + 1].w = 0;

                host_q[n * 4 + 2].x = grid_x * grid_size + 0.5 * grid_size;
                host_q[n * 4 + 2].y = grid_y * grid_size;
                host_q[n * 4 + 2].z = grid_z * grid_size + 0.5 * grid_size;
                host_q[n * 4 + 2].w = 0;

                host_q[n * 4 + 3].x = grid_x * grid_size;
                host_q[n * 4 + 3].y = grid_y * grid_size + 0.5 * grid_size;
                host_q[n * 4 + 3].z = grid_z * grid_size + 0.5 * grid_size;
                host_q[n * 4 + 3].w = 0;

                for (int i = 0; i < 4; i++)
                {
                    host_mol[n * 4 + i].set((MOLECULES) DEFAULT);

                    // Rms for Maxwell disftibution is (3kT/m)**0.5, for rand is 1
                    float velocity_coeff = sqrt((3 * K_B * T_INIT) / host_mol[n].M);

                    host_v[n * 4 + i].x = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 4 + i].y = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 4 + i].z = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 4 + i].w = 1;
                }
            }
}

void generate_bcc()
{
    int limit = 8; //(int) (cbrtf(N / 4));
    float grid_size = (2 * cell_size / limit);

    for (int grid_x = 0; grid_x < limit; grid_x++)
        for (int grid_y = 0; grid_y < limit; grid_y++)
            for (int grid_z = 0; grid_z < limit; grid_z++)
            {
                int n = grid_x * limit * limit + grid_y * limit + grid_z;

                host_q[n * 2].x = grid_x * grid_size;
                host_q[n * 2].y = grid_y * grid_size;
                host_q[n * 2].z = grid_z * grid_size;
                host_q[n * 2].w = 0;

                host_q[n * 2 + 1].x = grid_x * grid_size + 0.5 * grid_size;
                host_q[n * 2 + 1].y = grid_y * grid_size + 0.5 * grid_size;
                host_q[n * 2 + 1].z = grid_z * grid_size + 0.5 * grid_size;
                host_q[n * 2 + 1].w = 0;

                for (int i = 0; i < 2; i++)
                {
                    host_mol[n * 2 + i].set((MOLECULES) DEFAULT);

                    // Rms for Maxwell disftibution is (3kT/m)**0.5, for rand is 1
                    float velocity_coeff = sqrt((3 * K_B * T_INIT) / host_mol[n].M);

                    host_v[n * 2 + i].x = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 2 + i].y = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 2 + i].z = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 2 + i].w = 1;
                }
            }
}

void generate_dc()
{
    int limit = 4; //(int) (cbrtf(N / 4));
    float grid_size = (2 * cell_size / limit);

    for (int grid_x = 0; grid_x < limit; grid_x++)
        for (int grid_y = 0; grid_y < limit; grid_y++)
            for (int grid_z = 0; grid_z < limit; grid_z++)
            {
                int n = grid_x * limit * limit + grid_y * limit + grid_z;

                host_q[n * 8].x = grid_x * grid_size;
                host_q[n * 8].y = grid_y * grid_size;
                host_q[n * 8].z = grid_z * grid_size;
                host_q[n * 8].w = 0;

                host_q[n * 8 + 1].x = grid_x * grid_size + 0.5 * grid_size;
                host_q[n * 8 + 1].y = grid_y * grid_size + 0.5 * grid_size;
                host_q[n * 8 + 1].z = grid_z * grid_size;
                host_q[n * 8 + 1].w = 0;

                host_q[n * 8 + 2].x = grid_x * grid_size + 0.5 * grid_size;
                host_q[n * 8 + 2].y = grid_y * grid_size;
                host_q[n * 8 + 2].z = grid_z * grid_size + 0.5 * grid_size;
                host_q[n * 8 + 2].w = 0;

                host_q[n * 8 + 3].x = grid_x * grid_size;
                host_q[n * 8 + 3].y = grid_y * grid_size + 0.5 * grid_size;
                host_q[n * 8 + 3].z = grid_z * grid_size + 0.5 * grid_size;
                host_q[n * 8 + 3].w = 0;

                host_q[n * 8 + 4].x = grid_x * grid_size + 0.75 * grid_size;
                host_q[n * 8 + 4].y = grid_y * grid_size + 0.75 * grid_size;
                host_q[n * 8 + 4].z = grid_z * grid_size + 0.75 * grid_size;
                host_q[n * 8 + 4].w = 0;

                host_q[n * 8 + 5].x = grid_x * grid_size + 0.75 * grid_size;
                host_q[n * 8 + 5].y = grid_y * grid_size + 0.25 * grid_size;
                host_q[n * 8 + 5].z = grid_z * grid_size + 0.25 * grid_size;
                host_q[n * 8 + 5].w = 0;

                host_q[n * 8 + 6].x = grid_x * grid_size + 0.25 * grid_size;
                host_q[n * 8 + 6].y = grid_y * grid_size + 0.75 * grid_size;
                host_q[n * 8 + 6].z = grid_z * grid_size + 0.25 * grid_size;
                host_q[n * 8 + 6].w = 0;

                host_q[n * 8 + 7].x = grid_x * grid_size + 0.25 * grid_size;
                host_q[n * 8 + 7].y = grid_y * grid_size + 0.25 * grid_size;
                host_q[n * 8 + 7].z = grid_z * grid_size + 0.75 * grid_size;
                host_q[n * 8 + 7].w = 0;

                for (int i = 0; i < 8; i++)
                {
                    host_mol[n * 8 + i].set((MOLECULES) DEFAULT);

                    // Rms for Maxwell disftibution is (3kT/m)**0.5, for rand is 1
                    float velocity_coeff = sqrt((3 * K_B * T_INIT) / host_mol[n].M);

                    host_v[n * 8 + i].x = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 8 + i].y = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 8 + i].z = velocity_coeff * 2 * ((float) rand() / RAND_MAX - 0.5);
                    host_v[n * 8 + i].w = 1;
                }
            }
}

void get_params(float4 e, float4& params)
{
    float P = 0, V = 0, T = 0;

    V = 8 * cell_size * cell_size * cell_size;
    T = e.y * 2 / (3 * N * K_B);
    P = N * K_B * T / V + e.w / (6 * V); // TODO: Move fix (1/2) for virial to get_virial

#ifdef LJ_CUT // Cutoff set to half the box length
    P += (16 / 3) * M_PI * ro * ro * ((2 / 3) * powf(cell_size, -9) - powf(cell_size, -3));
#endif

    params = {P, V, T, 0};
}

void snapshot(ofstream& particles, ofstream& energy, ofstream& parameters)
{
    float E = 0, E_KIN = 0, E_POT = 0, VIRIAL = 0;
    if (SNAP_XYZ)
    {
        cudaMemcpy(host_q, device_q, sizeof(float4) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(host_v, device_v, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(host_e, device_e, sizeof(float4) * N, cudaMemcpyDeviceToHost);

    if (SNAP_XYZ)
        particles << N << endl << endl;

    for (int i = 0; i < N; i++)
    {
        if (SNAP_XYZ)
        {
            parameters << setprecision(15);
            particles << host_q[i].x << " ";
            particles << host_q[i].y << " ";
            particles << host_q[i].z << " ";

            particles << host_v[i].x << " ";
            particles << host_v[i].y << " ";
            particles << host_v[i].z << endl;
        }

        E += host_e[i].z;
        E_POT += host_e[i].x;
        E_KIN += host_e[i].y;
        VIRIAL += host_e[i].w;
    }

#ifdef LJ_CUT // Cutoff set to half the box length
    E_POT += (8 / 3) * M_PI * ro * ((1 / 3) * powf(cell_size, -9) - powf(cell_size, -3));
#endif

    energy << step << ",";
    energy << E_POT << ",";
    energy << E_KIN << ",";
    energy << E << ",";
    energy << VIRIAL << endl;

    get_params({E_POT, E_KIN, E, VIRIAL}, params);
    parameters << setprecision(15);
    parameters << params.x << ",";
    parameters << params.y << ",";
    parameters << params.z << endl;

    std::cout << "Step: " << step << " ";
    std::cout << setprecision(15) << "Energy: " << E << " ";
    std::cout << "Pressure: " << params.x << " ";
    std::cout << "Temperature: " << params.z << endl;
}

void load_dump(string name)
{
    ifstream dump(name);

    for (int i = 0; i < N; i++)
    {
        dump >> host_q[i].x;
        dump >> host_q[i].y;
        dump >> host_q[i].z;

        dump >> host_v[i].x;
        dump >> host_v[i].y;
        dump >> host_v[i].z;

        dump >> host_e[i].x;
        dump >> host_e[i].y;
        dump >> host_e[i].z;
        dump >> host_e[i].w;

        dump >> host_mol[i].Q;
        dump >> host_mol[i].M;
        dump >> host_mol[i].SIGMA;
        dump >> host_mol[i].EPSILON;
    }

    dump.close();
}

void create_dump(string name)
{
    ofstream dump(name);

    cudaMemcpy(host_q, device_q, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_v, device_v, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_e, device_e, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_mol, device_mol, sizeof(float4) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        dump << host_q[i].x << " ";
        dump << host_q[i].y << " ";
        dump << host_q[i].z << " ";

        dump << host_v[i].x << " ";
        dump << host_v[i].y << " ";
        dump << host_v[i].z << " ";

        dump << host_e[i].x << " ";
        dump << host_e[i].y << " ";
        dump << host_e[i].z << " ";
        dump << host_e[i].w << " ";

        dump << host_mol[i].Q << " ";
        dump << host_mol[i].M << " ";
        dump << host_mol[i].SIGMA << " ";
        dump << host_mol[i].EPSILON << endl;
    }

    dump.close();
}

int main()
{
    std::cout << ro << endl;
    cudaMalloc(&device_q, sizeof(float4) * N);
    cudaMalloc(&device_v, sizeof(float4) * N);
    cudaMalloc(&device_e, sizeof(float4) * N);
    cudaMalloc(&device_mol, sizeof(Molecule) * N);

    host_q = (float4*) malloc(sizeof(float4) * N);
    host_v = (float4*) malloc(sizeof(float4) * N);
    host_e = (float4*) malloc(sizeof(float4) * N);
    host_mol = (Molecule*) malloc(sizeof(Molecule) * N);

    //generate();
    //generate_fcc();
    //generate_bcc();
    generate_dc();
    // load_dump("research/glass/start.dat");
    // create_dump("glass.dat");

    ofstream particles("research/eam/eam.xyz");
    ofstream energy("research/eam/energy.csv");
    energy << "t,Potential,Kinetic,Total,Virial" << endl;
    ofstream velocity("data/gpuv.csv");
    velocity << "vx,vy,vz,v" << endl;
    ofstream parameters("research/eam/params.csv");
    parameters << "P,V,T" << endl;

    cudaMemcpy(device_q, host_q, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, host_v, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_e, host_e, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mol, host_mol, sizeof(Molecule) * N, cudaMemcpyHostToDevice);

    bool snap = false;
    for (step = 0; step < total_steps; step++)
    {
        if ((step % snap_steps == 0) )// || (step % thermo_steps == 0))
            snap = true;
#ifndef __INTELLISENSE__
#ifdef EAM_POTENTIAL
        evolve_densities<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(device_q, N);
#endif
        evolve<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(device_q, device_v, device_mol, N, dt, device_e, snap);
#endif
        if ((step % snap_steps == 0) && (step > 0))
            snapshot(particles, energy, parameters);

        //  if ((step % thermo_steps == 0) && (step < total_steps))
        //      thermostat_scale();

        snap = false;
    }

    create_dump("research/eam/start.dat");

    cudaMemcpy(host_v, device_v, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    float v2 = 0;
    for (int i = 0; i < N; i++)
    {
        v2 = host_v[i].x * host_v[i].x + host_v[i].y * host_v[i].y + host_v[i].z * host_v[i].z;
        velocity << host_v[i].x << ",";
        velocity << host_v[i].y << ",";
        velocity << host_v[i].z << ",";
        velocity << v2 << endl;
    }

    particles.close();
    energy.close();
    velocity.close();
    parameters.close();

    cudaFree(device_q);
    cudaFree(device_v);
    cudaFree(device_e);
    cudaFree(device_mol);
    delete (host_q);
    delete (host_v);
    delete (host_e);
    delete (host_mol);

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
