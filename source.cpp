#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>

using namespace std;

#define N 512
#define file "data.xyz"
#define diagnostics "diag.csv"

float q[N][3];
float v[N][3];
float a[N][3];
float T = 0, P = 0, E = 0;
int step = 0, total = 10000;
int t = 0, dt = 1, t_diag = 5, t_snap = 5;

ofstream dump;
ofstream diag;

void generate();

void interract(int i, int j);

void evolve();

void diagnose();

void snapshot();

int main()
{
    dump.open(file, ios::out);
    diag.open(diagnostics, ios::out);
    diag << "Particles,"
         << "Step,"
         << "Potential,"
         << "Kitetic,"
         << "Total" << endl;

    cout << "Simulating " << N << " bodies for " << total << " steps" << endl;

    generate();
    snapshot();
    // diagnose();

    //total = -1;
    while (step < total)
    {
        evolve();
        // cout << step << endl;
    }

    dump.close();
    diag.close();
}

void generate()
{
    for (int i = 0; i < 8; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                int n = i * 8 * 8 + j * 8 + k;
                q[n][0] = 10*i;
                q[n][1] = 10*j;
                q[n][2] = 10*k;
                v[n][0] = 0.0;
                v[n][1] = 0.0;
                v[n][2] = 0.0;
            }
        }
    }
}

void interract(int i, int j)
{
    if(i ==j)
      return;
    float r[3] = {0.0, 0.0, 0.0};
    r[0] = q[i][0] - q[j][0];
    r[1] = q[i][1] - q[j][1];
    r[2] = q[i][2] - q[j][2];

    float k = pow((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), 1.5);
    float p = pow((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), -0.5);
    P += p;

    for (int iter = 0; iter < 3; iter++)
    {
        a[i][iter] += r[iter] / k;
        a[j][iter] += -r[iter] / k;
    }

    // TODO: Calculate P here
}

void evolve()
{
    for (int i = 0; i < N; i++)
    {
      a[i][0] = 0.0;
      a[i][1] = 0.0;
      a[i][2] = 0.0;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            interract(i, j);
        }
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            v[i][j] += a[i][j] * dt;
            q[i][j] += v[i][j] * dt + a[i][j] * dt *dt / 2;
        }
    }
    t+=dt;

    if ((t % t_diag) == 0)
        diagnose();

    if ((t % t_snap) == 0)
        snapshot();

    E = P = T = 0;
    step++;
}

void diagnose()
{
    for (int i = 0; i < N; i++)
    {
        float vel2 = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
        T += vel2 / 2;
    }
    E=T+P;
    cout << "Step: " << step << " P=" << P << " T=" << T << " E=" << E << endl;
}

void snapshot()
{
    dump << N << endl << endl;

    for (int i = 0; i < N; i++)
    {
        dump << q[i][0] << " " << q[i][1] << " " << q[i][2] << " " << endl;
    }
    diag << N << "," << step << "," << P << "," << T << "," << E << endl;
}
