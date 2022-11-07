#include <cassert>

#include "dense_layer.h"
#include "linalg/linalg.h"

using namespace std;
using namespace seal;

DenseLayer::DenseLayer(shared_ptr<CKKSManager> ckks, int in_sz, int out_sz) {
    this->ckks = ckks;

    this->in_sz = in_sz;
    this->out_sz = out_sz;
    // Random init
    bias = get_random_vector(out_sz);
    for (int i = 0; i < out_sz; ++i) {
        weight.push_back(get_random_vector(in_sz));
    }
    weight_diags = diagonals_from_matrix(weight);
}

DenseLayer::DenseLayer(shared_ptr<CKKSManager> ckks, int in_sz, int out_sz, const vector<double>& raw_w, const vector<double>& raw_b, bool rescaled, int divide, double add, int block_sz, int nb_slots) {
    this->ckks = ckks;

    this->in_sz = in_sz;
    this->out_sz = out_sz;

    // w: [real_out_sz x real_in_sz], b: [real_out_sz]
    // padding
    int real_out_sz = raw_b.size();
    int real_in_sz = raw_w.size() / real_out_sz;

    cout << "Loading a " << real_out_sz << " x " << real_in_sz << " weight matrix";
    cout << " into a " << out_sz << " x " << in_sz << " weight matrix." << endl;

    if (real_out_sz > out_sz || real_in_sz > in_sz) {
        throw invalid_argument("impossible to pad like this");
    }

    bias = std::vector<double>(out_sz, 0);
    for (int i = 0; i < real_out_sz; ++i) {
        bias[i] = raw_b[i];
        if (rescaled) bias[i] = bias[i]/divide + add; // rescale
    }

    for (int i = 0; i < out_sz; ++i) {
        weight.push_back(std::vector<double>(in_sz, 0));
    }

    for (int i = 0; i < real_out_sz; ++i) {
        for (int j = 0; j < real_in_sz; ++j) {
            int idx = i*real_in_sz+j;
            weight[i][j] = raw_w[idx];
            if (rescaled) weight[i][j] /= divide; // rescale
        }
    }

    // To make batched NNs work, we simply need to make batched weight diags
    // of size nb_slots (repeated in each block)
    weight_diags = diagonals_from_matrix(weight);

    weight_diags_batched.resize(weight_diags.size());    

    int nb_blocks = nb_slots / block_sz;
    for (int i = 0; i < out_sz; i++) {
        weight_diags_batched[i].resize(nb_slots, 0);
        
        for (int idx = 0; idx < nb_blocks; idx++) {
            int start_idx = block_sz*idx;
            for (int j = 0; j < in_sz; j++) {
                weight_diags_batched[i][start_idx+j] = weight_diags[i][j];
            }
        }
    }

    // We also need to make batched bias
    bias_batched.resize(nb_slots, 0);
    for (int idx = 0; idx < nb_blocks; idx++) {
        int start_idx = block_sz*idx;
        for (int j = 0; j < out_sz; j++) {
            bias_batched[start_idx+j] = bias[j];
        }
    }

}

// Homomorphic inference
void DenseLayer::forward(Ciphertext& input, Ciphertext& dest) {
    // Encode the diag as p so the result is (Dp) / p = D
    double D = input.scale();
    uint64_t p = ckks->get_modulus(input, 1);

    mvp_hybrid_ctx(ckks->galois_keys, 
    *(ckks->evaluator), *(ckks->encoder), out_sz, in_sz, weight_diags_batched, input, dest, p);

    Plaintext bias_plain;
    ckks->encoder->encode(bias_batched, dest.parms_id(), dest.scale(), bias_plain);
    ckks->evaluator->add_plain_inplace(dest, bias_plain);
    ckks->evaluator->rescale_to_next_inplace(dest);
    assert(fabs(dest.scale() - D) < 1e-3); // sanity check
    
    if (this->activation != nullptr) {
        // This should also ~preserve the scale
        this->activation->forward(dest, dest);
    }
}

// Standard inference
void DenseLayer::forward(vector<double>& input, vector<double>& dest) {
    assert(input.size() == in_sz);
    
    dest = mvp_hybrid_ptx(weight_diags, input); 
    transform(dest.begin(), dest.end(), bias.begin(), dest.begin(), plus<double>());
    if (this->activation != nullptr) {
        this->activation->forward(dest, dest);
    }
}