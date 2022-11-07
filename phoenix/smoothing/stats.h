#pragma once 

#include <vector>
#include <cassert>
#include <memory>
#include "utils/ckks_manager.h"
#include "hcephes.h"

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

inline int get_min_n_winner_to_not_abstain(int N, double tau, double alpha) {
    for (int n_winner = N; n_winner >= 0; n_winner--) {
        double pval = hcephes_bdtrc(n_winner-1, N, tau);
        if (pval > alpha) return n_winner+1; // previous was the min valid
    }
    assert(false);
    return -1; // must never happen
}

inline double get_radius(double sigma, double tau) {
    return sigma * hcephes_ndtri(tau);
}