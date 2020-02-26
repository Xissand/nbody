#include "cuda_runtime.h"
//#include "kernel.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <iostream>
#include <stdio.h>

#define PERIODIC_BOUNDARIES
//#define COULOMB_POTENTIAL
#define LENNARD_JONES_POTENTIAL

#define BLOCK_SIZE 128
const int cell_size = 8;
#define N 4096

float4* device_q;
float4* device_v;
float4* host_q;
float4* host_v;
float3* device_e;
float3* host_e;

__device__ void interract(float4 qi, float4 qj, float3& ai, float& p)
{
    float3 r;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

    if (r.x == 0 && r.y == 0 && r.z == 0)
        return;

// Periodic boundary checks
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

    float k = 0;
    // apparently powf can be more expensive, not sure

#ifdef LENNARD_JONES_POTENTIAL
    // p=r_12-r_6
    // f=-r_14+r_8

    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r4 * r4;

    k = (-(0.5f) + (1.0f / r6)) * 12 / r8;
    p += (-(1.0f) + (1.0f / r6)) * 1 / r6;
#endif

#ifdef COULOMB_POTENTIAL
    if (r2 < 0.0001)
        r2 = 0.0001;

    float r_mod = sqrtf(r2);
    float r3 = r_mod * r_mod * r_mod;

    k += 1.0f / r3;
    p += 1.0f / r_mod;
#endif

    ai.x += r.x * k;
    ai.y += r.y * k;
    ai.z += r.z * k;
    // printf("%f\n", r2);

    // printf("%f %f %f\n", ai.x, ai.y, ai.z);
}

__global__ void evolve(float4* d_q, float4* d_v, int N_BODIES, float dt, float3* d_e)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float4 shared_q[BLOCK_SIZE];
    float4 q = d_q[j];
    float4 v = d_v[j];
    float3 e = {0.0f, 0.0f, 0.0f};
    float3 currA = {0.0f, 0.0f, 0.0f};
    float e_pot = 0;

    for (int i = 0; i < N_BODIES; i += BLOCK_SIZE)
    {
        shared_q[threadIdx.x] = d_q[i + threadIdx.x];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            interract(q, shared_q[k], currA, e_pot);
        }
        __syncthreads();
    }

    v.x += currA.x * dt;
    v.y += currA.y * dt;
    v.z += currA.z * dt;

    float e_kin = (v.x * v.x + v.y * v.y + v.z * v.z) / 2.0f;

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

    float currA2 = currA.x * currA.x + currA.y * currA.y + currA.z * currA.z;
    e.x += e_pot / 2;
    e.y += e_kin;
    e.z += e_pot / 2 + e_kin;

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

                host_e[n].x = 0;
                host_e[n].y = 0;
                host_e[n].z = 0;
            }
    // TODO: implement generation for any number of particles
    // TODO: implement velocity generation based on temperature
}

int main()
{
    float dt = 0.001;
    int total_steps = 100000;
    int snap_steps = 5;
    float E = 0, E_KIN = 0, E_POT = 0;

    cudaMalloc(&device_q, sizeof(float4) * N);
    cudaMalloc(&device_v, sizeof(float4) * N);
    cudaMalloc(&device_e, sizeof(float3) * N);

    host_q = (float4*) malloc(sizeof(float4) * N);
    host_v = (float4*) malloc(sizeof(float4) * N);
    host_e = (float3*) malloc(sizeof(float3) * N);

    generate();

    FILE* fp;
    fp = fopen("particles.xyz", "w");
    FILE* fp2;
    fp2 = fopen("gpue.csv", "w");
    fprintf(fp2, "t,Potential,Kinetic,Total\n");
    FILE* fp3;
    fp3 = fopen("gpuv.csv", "w");
    fprintf(fp3, "vx,vy,vz,v\n");

    cudaMemcpy(device_q, host_q, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, host_v, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_e, host_e, sizeof(float3) * N, cudaMemcpyHostToDevice);

    for (int step = 0; step < total_steps; step++)
    {
        evolve<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(device_q, device_v, N, dt, device_e);
        if (step % snap_steps == 0)
        {
            E = E_POT = E_KIN = 0.0f;
            cudaMemcpy(host_q, device_q, sizeof(float4) * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_e, device_e, sizeof(float3) * N, cudaMemcpyDeviceToHost);
            fprintf(fp, "%d\n\n", N);
            for (int i = 0; i < N; i++)
            {
                fprintf(fp, "%f %f %f\n", host_q[i].x, host_q[i].y, host_q[i].z);
                E += host_e[i].z;
                E_POT += host_e[i].x;
                E_KIN += host_e[i].y;
            }
            fprintf(fp2, "%d,%f,%f,%f\n", step, E_POT, E_KIN, E);
        }
    }

    cudaMemcpy(host_v, device_v, sizeof(float4) * N, cudaMemcpyDeviceToHost);
    float v2 = 0;
    for (int i = 0; i < N; i++)
    {
        v2 = host_v[i].x * host_v[i].x + host_v[i].y * host_v[i].y + host_v[i].z * host_v[i].z;
        fprintf(fp3, "%f,%f,%f,%f\n", host_v[i].x, host_v[i].y, host_v[i].z, v2);
    }

    fclose(fp);
    fclose(fp2);
    fclose(fp3);

    cudaFree(device_q);
    cudaFree(device_v);
    cudaFree(device_e);
    delete (host_q);
    delete (host_v);
    delete (host_e);

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
