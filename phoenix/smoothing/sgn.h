#pragma once

#include <mutex>

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif
#include "utils/ckks_manager.h"

using namespace std;
using namespace seal;

class SgnEvaluator {
public:
    SgnEvaluator(bool VERBOSE, shared_ptr<CKKSEncoder> encoder, double mul_last=1) {
        this->VERBOSE = VERBOSE;
        this->mul_last = mul_last;

        // Precompute coeffs and coeffs_last by dividing by scale
        f4_coeffs_last.resize(10, 0);
        g4_coeffs_last.resize(10, 0);
        for (int i : {1, 3, 5, 7, 9}) {
            f4_coeffs[i] /= f4_scale;
            f4_coeffs_last[i] = f4_coeffs[i] * mul_last;

            g4_coeffs[i] /= g4_scale;
            g4_coeffs_last[i] = g4_coeffs[i] * mul_last;
        }
    }

    // Evaluates f4^(df) o g4^(dg) at x (Ciphertext)
    void sgn(int dg, int df, Ciphertext& x, Ciphertext& dest, shared_ptr<CKKSManager> ckks);

    // Evaluates f4^(df) o g4^(dg) at x (double)
    double sgn_plain(int dg, int df, double x);
private:
    bool VERBOSE;
    double mul_last;

    vector<double> f4_coeffs = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
    vector<double> f4_coeffs_last;
    // should be divided by (1 << 7)
    int f4_scale = (1 << 7);

    vector<double> g4_coeffs = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
    vector<double> g4_coeffs_last;
    // should be divided by (1 << 10)
    int g4_scale = (1 << 10);

    // Evaluates the odd degree9 polynomial given by the coefficients in a[]
    // NOTE: might modswitch both a/x (won't change anything else though)
    void eval_odd_deg9_poly(vector<double>& a, Ciphertext& x, Ciphertext& dest, shared_ptr<CKKSManager> ckks);
};