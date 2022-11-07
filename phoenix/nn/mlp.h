#pragma once

#include <vector>
#include <cassert>
#include "linalg/linalg.h"
#include "nn/dense_layer.h"
#include "utils/ckks_manager.h"
#include "utils/rapidcsv.h"

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

class MLP {
public:
    MLP(bool VERBOSE, std::shared_ptr<CKKSManager> ckks, int block_sz, int nb_slots, int input_size, const std::vector<int>& dense_sizes, const std::string& model_path, bool rescaled, double divide=0, double add=0);

    // Homomorphic inference
    void forward(seal::Ciphertext& input, seal::Ciphertext& dest);

    // Standard inference
    void forward(const std::vector<double>& input, std::vector<double>& dest);

private:
    std::shared_ptr<CKKSManager> ckks;

    std::unique_ptr<rapidcsv::Document> model_csv;

    bool VERBOSE; 
    
    bool rescaled; // if we forwarded (divide, add) to rescale the logits
    int block_sz;
    int nb_slots;
    int input_size;
    int nb_classes;
    std::vector<DenseLayer> layers; // square activation after each
};
