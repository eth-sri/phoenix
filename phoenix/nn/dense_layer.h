#pragma once

#include <iostream>
#include <vector>

#include "utils/ckks_manager.h"
#include "utils/helpers.h"
#include "linalg/linalg.h"

#include "learnable_square.h"

class DenseLayer {
public:
    DenseLayer(std::shared_ptr<CKKSManager> ckks, int in_sz, int out_sz);
    DenseLayer(std::shared_ptr<CKKSManager> ckks, int in_sz, int out_sz, const std::vector<double>& raw_w, const std::vector<double>& raw_b, bool rescaled, int divide, double add, int block_sz, int nb_slots);

    // Homomorphic inference
    void forward(seal::Ciphertext& input, seal::Ciphertext& dest);

    // Standard inference
    void forward(std::vector<double>& input, std::vector<double>& dest);

    int get_out_sz() {return out_sz;}

    void set_activation(std::vector<double> a, std::vector<double> b) {
        activation = std::make_shared<LearnableSquare>(ckks, a, b);
    }

private:
    std::shared_ptr<CKKSManager> ckks;

    int in_sz, out_sz;
    std::vector<std::vector<double>> weight;
    std::vector<std::vector<double>> weight_diags;
    std::vector<std::vector<double>> weight_diags_batched;
    std::vector<double> bias;
    std::vector<double> bias_batched;

    std::shared_ptr<LearnableSquare> activation;
};