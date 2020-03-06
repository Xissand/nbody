#pragma once

enum MOLECULES
{
    PROTON,
    ELECTRON,
    DEFAULT
};

struct Molecule {
    float Q;
    float M;
    float SIGMA;
    float EPSILON;
    void set(MOLECULES type)
    {
        switch (type)
        {
        case PROTON:
            Q = 1;
            M = 1;
            SIGMA = 1;
            EPSILON = 1;
            break;
        case ELECTRON:
            Q = 1;
            M = 1;
            SIGMA = 1;
            EPSILON = 1;
            break;
        case DEFAULT:
            Q = 1;
            M = 1;
            SIGMA = 1;
            EPSILON = 1e3;
            break;
        default:
            Q = 1;
            M = 1;
            SIGMA = 1;
            EPSILON = 1;
            break;
        }
    }
    Molecule(MOLECULES type)
    {
        set(type);
    }
};