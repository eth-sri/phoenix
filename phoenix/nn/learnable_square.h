#pragma once

#include <iostream>
#include <vector>
#include "utils/ckks_manager.h"

class LearnableSquare {
public:
    LearnableSquare(std::shared_ptr<CKKSManager> ckks, std::vector<double> a, std::vector<double> b);

    // Homomorphic inference
    void forward(seal::Ciphertext& input, seal::Ciphertext& dest);

    // Standard inference
    void forward(std::vector<double>& input, std::vector<double>& dest);

private:
    std::shared_ptr<CKKSManager> ckks;
    std::vector<double> a;
    std::vector<double> b;
};