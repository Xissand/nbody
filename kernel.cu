#include "cuda_runtime.h"
//#include "kernel.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define BLOCK_SIZE 256
const int cell_size = 250;
#define N 4096

float4* device_q;
float4* device_v;
float4* host_q;
float4* host_v;
float3* device_e;
float3* host_e;

float E = 0, E_KIN = 0, E_POT = 0;

__device__ void interract(float4 qi, float4 qj, float3& ai, float& p)
{
    float3 r;
    r.x = qi.x - qj.x;
    r.y = qi.y - qj.y;
    r.z = qi.z - qj.z;

    float r2 = r.x * r.x + r.y * r.y + r.z * r.z;
    if (r2 < 0.01)
        r2 = 0.01;

    // apparently powf can be more expensive, not sure
    float r_mod = sqrtf(r2);
    float r3 = r_mod * r_mod * r_mod;

    float k = 1.0f / r3;
    p += 1.0f / r_mod;

    ai.x += r.x * k;
    ai.y += r.y * k;
    ai.z += r.z * k;
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
    // q.w = e_pot / 2.0f + e_kin;

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

    e.x += e_pot / 2;
    e.y += e_kin;
    e.z += e_pot / 2 + e_kin;

    /*if (q.x > cell_size)
    {
            q.x -= 2*(q.x-cell_size);
            v.x = -v.x;
    }
    if (q.x < -cell_size)
    {
            q.x += 2*(-q.x - cell_size);
            v.x = -v.x;
    }
    if (q.y > cell_size)
    {
            q.y -= 2*(q.y-cell_size);
            v.y = -v.y;
    }
    if (q.y < -cell_size)
    {
            q.y += 2*(-q.y - cell_size);
            v.y = -v.y;
    }
    if (q.z > cell_size)
    {
            q.z -= 2*(q.z-cell_size);
            v.z = -v.z;
    }
    if (q.z < -cell_size)
    {
            q.z += 2*(-q.z - cell_size);
            v.z = -v.z;
    }
*/

    __syncthreads();

    // E_POT += e_pot;
    // E_KIN += e_kin;
    d_q[j] = q;
    d_v[j] = v;
    d_e[j] = e;
}

int main()
{
    cudaMalloc(&device_q, sizeof(float4) * N);
    cudaMalloc(&device_v, sizeof(float4) * N);
    cudaMalloc(&device_e, sizeof(float3) * N);

    host_q = (float4*) malloc(sizeof(float4) * N);
    host_v = (float4*) malloc(sizeof(float4) * N);
    host_e = (float3*) malloc(sizeof(float3) * N);

    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 16; j++)
            for (int k = 0; k < 16; k++)
            {
                host_q[256 * i + 16 * j + k].x = i * 10 + rand() % 10 - 80;
                host_q[256 * i + 16 * j + k].y = j * 10 + rand() % 10 - 80;
                host_q[256 * i + 16 * j + k].z = k * 10 + rand() % 10 - 80;
                host_q[256 * i + 16 * j + k].w = 0;

                host_v[256 * i + 16 * j + k].x = 0;
                host_v[256 * i + 16 * j + k].y = 0;
                host_v[256 * i + 16 * j + k].z = 0;
                host_v[256 * i + 16 * j + k].w = 1;

                host_e[256 * i + 16 * j + k].x = 0;
                host_e[256 * i + 16 * j + k].y = 0;
                host_e[256 * i + 16 * j + k].z = 0;
            }

    FILE* fp;
    fp = fopen("particles.xyz", "w");
    FILE* fp2;
    fp2 = fopen("gpue.csv", "w");
    fprintf(fp2, "t,Potential,Kinetic,Total\n");

    cudaMemcpy(device_q, host_q, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, host_v, sizeof(float4) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(device_e, host_e, sizeof(float3) * N, cudaMemcpyHostToDevice);

    for (int t = 0; t < 20000; t++)
    {
        evolve<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(device_q, device_v, N, 0.1, device_e);
        if (t % 5 == 0)
        {
            E = E_POT = E_KIN = 0.0f;
            cudaMemcpy(host_q, device_q, sizeof(float4) * N, cudaMemcpyDeviceToHost);
            cudaMemcpy(host_e, device_e, sizeof(float3) * N, cudaMemcpyDeviceToHost);
            fprintf(fp, "%d\n\n", N);
            for (int i = 0; i < N; i++)
            {
                fprintf(fp, "%f %f %f\n", host_q[i].x, host_q[i].y, host_q[i].z);
                E+=host_e[i].z;
                E_POT+=host_e[i].x;
                E_KIN+=host_e[i].y;
            }
            fprintf(fp2, "%d,%f,%f,%f\n", t, E_POT, E_KIN, E);
            // printf("energy: %f\n", E);
        }
    }

    fclose(fp);
    fclose(fp2);

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
