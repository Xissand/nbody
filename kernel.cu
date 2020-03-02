#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "molecules.h"
#include "units.h"
#include <cmath>
#include <fstream>
#include <iostream>

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

    if (r.x == 0 && r.y == 0 && r.z == 0)
        return;

#ifdef PERIODIC_BOUNDARIES
    if (r.x > cell_size)
        r.x -= 2 * cell_size;
    if (r.x <= -cell_size)
        r.x += 2 * cell_size;
    if (r.y > cell_size)
        r.y -= 2 * cell_size;
    if (r.y <= -cell_size)
        r.y += 2 * cell_size;
    if (r.z > cell_size)
        r.z -= 2 * cell_size;
    if (r.z <= -cell_size)
        r.z += 2 * cell_size;
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;

#ifdef LENNARD_JONES_POTENTIAL
    r2 = r2 / (moli.SIGMA * moli.SIGMA);
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r4 * r4;

    float k = (-(0.5f) + (1.0f / r6)) * 12 / r8;
    coeff = 4 * moli.EPSILON;
#endif

#ifdef COULOMB_POTENTIAL
    if (r2 < 0.0001)
        r2 = 0.0001;

    float r_mod = sqrtf(r2);
    float r3 = r_mod * r_mod * r_mod;

    float k += 1.0f / r3;
#endif

    fi.x += r.x * coeff * k;
    fi.y += r.y * coeff * k;
    fi.z += r.z * coeff * k;
}

__device__ void get_virial(float4 qi, float4 qj, Molecule moli, float4& e)
{
    float3 r;
    float coeff = 1;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

    if (r.x == 0 && r.y == 0 && r.z == 0)
        return;

#ifdef PERIODIC_BOUNDARIES
    if (r.x > cell_size)
        r.x -= 2 * cell_size;
    if (r.x <= -cell_size)
        r.x += 2 * cell_size;
    if (r.y > cell_size)
        r.y -= 2 * cell_size;
    if (r.y <= -cell_size)
        r.y += 2 * cell_size;
    if (r.z > cell_size)
        r.z -= 2 * cell_size;
    if (r.z <= -cell_size)
        r.z += 2 * cell_size;
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;

#ifdef LENNARD_JONES_POTENTIAL
    r2 = r2 / (moli.SIGMA * moli.SIGMA);
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r4 * r4;

    coeff = 4 * moli.SIGMA;
    float k = (-(0.5f) + (1.0f / r6)) * 12 / r8;
#endif

#ifdef COULOMB_POTENTIAL
    if (r2 < 0.0001)
        r2 = 0.0001;

    float r_mod = sqrtf(r2);
    float r3 = r_mod * r_mod * r_mod;

    float k += 1.0f / r3;
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

    if (r.x == 0 && r.y == 0 && r.z == 0)
        return;

#ifdef PERIODIC_BOUNDARIES
    if (r.x > cell_size)
        r.x -= 2 * cell_size;
    if (r.x <= -cell_size)
        r.x += 2 * cell_size;
    if (r.y > cell_size)
        r.y -= 2 * cell_size;
    if (r.y <= -cell_size)
        r.y += 2 * cell_size;
    if (r.z > cell_size)
        r.z -= 2 * cell_size;
    if (r.z <= -cell_size)
        r.z += 2 * cell_size;
#endif

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;

#ifdef LENNARD_JONES_POTENTIAL
    r2 = r2 / (moli.SIGMA * moli.SIGMA);
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    coeff = 4 * moli.EPSILON;

    p = (-(1.0f) + (1.0f / r6)) * coeff / r6;

    e.x += p / 2.0f;
    e.z += p / 2.0f;
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
    // float e_kin = (v.x * v.x + v.y * v.y + v.z * v.z) / 2.0f;

    q.x += v.x * dt;
    q.y += v.y * dt;
    q.z += v.z * dt;

#ifdef PERIODIC_BOUNDARIES
    if (q.x > cell_size)
        q.x -= 2 * cell_size;
    if ((-q.x) > cell_size)
        q.x += 2 * cell_size;
    if (q.y > cell_size)
        q.y -= 2 * cell_size;
    if ((-q.y) > cell_size)
        q.y += 2 * cell_size;
    if (q.z > cell_size)
        q.z -= 2 * cell_size;
    if ((-q.z) > cell_size)
        q.z += 2 * cell_size;
#endif

        // e.x += e_pot / 2;
        // e.y += e_kin;
        // e.z += e_kin;

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

void generate()
{
    int grid_size = (int) truncf(2 * cell_size / (cbrtf(N)));
    int limit = (int) truncf(cbrtf(N));

    for (int i = 0; i < limit; i++)
        for (int j = 0; j < limit; j++)
            for (int k = 0; k < limit; k++)
            {
                int n = limit * limit * i + limit * j + k;
                host_q[n].x = i * grid_size + rand() % grid_size - cell_size;
                host_q[n].y = j * grid_size + rand() % grid_size - cell_size;
                host_q[n].z = k * grid_size + rand() % grid_size - cell_size;
                host_q[n].w = 0;

                host_v[n].x = (rand() % 100) / 10 - 5;
                host_v[n].y = (rand() % 100) / 10 - 5;
                host_v[n].z = (rand() % 100) / 10 - 5;
                host_v[n].w = 1;

                host_e[n].x = 0; // Potential
                host_e[n].y = 0; // Kinetic
                host_e[n].z = 0; // Total
                host_e[n].w = 0; // Virial
            }
    for (int i = 0; i < N; i++)
    {

        host_mol[i].set((MOLECULES) DEFAULT);
    }
    // TODO: implement generation for any number of particles
    // TODO: implement velocity generation based on temperature
}

void get_params(float4 e, float4& params)
{
    float P = 0, V = 0, T = 0;

    V = cell_size * cell_size * cell_size;
    T = e.y * 1 * 2 / 3;
    P = N * 1 * T / V + e.w / (3 * V);

    params = {P, V, T, 0};
}

void snapshot(ofstream& particles, ofstream& energy, ofstream& parameters)
{
    float E = 0, E_KIN = 0, E_POT = 0, VIRIAL = 0;
    //cudaMemcpy(host_q, device_q, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_e, device_e, sizeof(float4) * N, cudaMemcpyDeviceToHost);

    particles << N << endl << endl;

    for (int i = 0; i < N; i++)
    {
        //particles << host_q[i].x << " ";
        //particles << host_q[i].y << " ";
        //particles << host_q[i].z << endl;

        E += host_e[i].z;
        E_POT += host_e[i].x;
        E_KIN += host_e[i].y;
        VIRIAL += host_e[i].w;
    }

    energy << step << ",";
    energy << E_POT << ",";
    energy << E_KIN << ",";
    energy << E << ",";
    energy << VIRIAL << endl;

    get_params({E_POT, E_KIN, E, VIRIAL}, params);
    parameters << params.x << ",";
    parameters << params.y << ",";
    parameters << params.z << endl;

    cout << "Step: " << step << " ";
    cout << "Energy: " << E << " ";
    cout << "Temperature: " << params.z << endl;
}

int main()
{
    cudaMalloc(&device_q, sizeof(float4) * N);
    cudaMalloc(&device_v, sizeof(float4) * N);
    cudaMalloc(&device_e, sizeof(float4) * N);
    cudaMalloc(&device_mol, sizeof(Molecule) * N);

    host_q = (float4*) malloc(sizeof(float4) * N);
    host_v = (float4*) malloc(sizeof(float4) * N);
    host_e = (float4*) malloc(sizeof(float4) * N);
    host_mol = (Molecule*) malloc(sizeof(Molecule) * N);

    generate();

    ofstream particles("particles.xyz");
    ofstream energy("gpue.csv");
    energy << "t,Potential,Kinetic,Total,Virial" << endl;
    ofstream velocity("gpuv.csv");
    velocity << "vx,vy,vz,v" << endl;
    ofstream parameters("gpuparam.csv");
    parameters << "P,V,T" << endl;

    cudaMemcpy(device_q, host_q, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, host_v, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_e, host_e, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mol, host_mol, sizeof(Molecule) * N, cudaMemcpyHostToDevice);

    bool snap = false;
    for (step = 0; step < total_steps; step++)
    {
        if (step % snap_steps == 0)
            snap = true;
#ifndef __INTELLISENSE__
        evolve<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(device_q, device_v, device_mol, N, dt, device_e, snap);
#endif
        if (step % snap_steps == 0)
            snapshot(particles, energy, parameters);
        snap = false;
    }

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
