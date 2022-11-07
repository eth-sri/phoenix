#include <cassert>

#include "mlp.h"
#include "linalg/linalg.h"

using namespace std;
using namespace seal;

MLP::MLP(bool VERBOSE, shared_ptr<CKKSManager> ckks, int block_sz, int nb_slots, int input_size, const vector<int>& dense_sizes, const string& model_path, bool rescaled, double divide, double add) {
    this->VERBOSE = VERBOSE;

    this->ckks = ckks;

    int n_layers = dense_sizes.size();
    this->block_sz = block_sz;
    this->nb_slots = nb_slots;
    this->input_size = input_size;
    this->nb_classes = dense_sizes[n_layers-1];
    this->rescaled = rescaled;

    model_csv = make_unique<rapidcsv::Document>(model_path, rapidcsv::LabelParams(-1, 0));

    int prev = input_size;
    for (int i = 0; i < n_layers; i++) {
        int curr = dense_sizes[i];

        stringstream w_key, b_key;
        w_key << "w" << i+1, b_key << "b" << i+1;

        // These have "real" sizes (dense_layers will pad)
        vector<double> weights = model_csv->GetRow<double>(w_key.str());
        vector<double> biases = model_csv->GetRow<double>(b_key.str());

        if (i == n_layers-1) {
            // Rescale the last layer so logits are in [0, 1]
            if (VERBOSE and this->rescaled and 0) cout << "Rescaling the last layer of the MLP with divide=" << divide << " and add=" << add << endl;
            DenseLayer dense(ckks, prev, curr, weights, biases, rescaled, divide, add, block_sz, nb_slots);
            layers.push_back(dense);
        } else {
            DenseLayer dense(ckks, prev, curr, weights, biases, false, 0, 0, block_sz, nb_slots);

            // Non-last layer must have activation
            stringstream act_a_key, act_b_key;
            act_a_key << "act" << i+1 << "_a";
            act_b_key << "act" << i+1 << "_b";
            vector<double> act_a = model_csv->GetRow<double>(act_a_key.str());
            vector<double> act_b = model_csv->GetRow<double>(act_b_key.str());
            assert(act_a.size()==1 && act_b.size() == 1);

            dense.set_activation(vector<double>(nb_slots, act_a[0]), vector<double>(nb_slots, act_b[0]));

            layers.push_back(dense);
        }
        prev = curr;
    }

}

// Homomorphic inference
// Scales chosen for errorless evaluation and to keep the scale the same (like sgn.cpp)

void MLP::forward(Ciphertext& input, Ciphertext& dest) {
    stringstream ss_time;
    auto t_forward_start = Clock::now();

    Ciphertext curr = input;
    double D = input.scale(); // maybe not init_scale but preserved
    
    // Forward all dense layers
    for (int i = 0; i < layers.size(); i++) {
        auto t_layer_start = Clock::now();
        
        layers[i].forward(curr, curr); // if activation is there it will also be applied
        // ~preserves the scale (up to floating point)

        if (i < layers.size()-1) {
            // we must homomorphically duplicate to get a "well rotatable" vector
            // (needed for next dense layer) 
            // done for the batched case!
            vector<double> mask(layers[i].get_out_sz(), 1);
            extend_mask_to_blocks(mask, nb_slots, block_sz);

            // Encode this at scale p so the result after rescale is (Dp) / p = D 
            uint64_t p = ckks->get_modulus(curr, 1);

            Plaintext mask_plain;
            ckks->encoder->encode(mask, curr.parms_id(), p, mask_plain);
            ckks->evaluator->multiply_plain_inplace(curr, mask_plain); // masked

            Ciphertext rotated;
            ckks->evaluator->rotate_vector(curr, -layers[i].get_out_sz(), ckks->galois_keys, rotated);
            ckks->evaluator->add_inplace(curr, rotated);
            ckks->evaluator->rescale_to_next_inplace(curr);
        }

        auto t_layer_end = Clock::now();
    }
    dest = std::move(curr);

    auto t_forward_end = Clock::now();
    if (VERBOSE) log_time(ss_time, "Forward done", t_forward_start, t_forward_end);
}

// Standard inference
void MLP::forward(const vector<double>& input, vector<double>& dest) {
    assert(input.size() == input_size);

    vector<double> curr = input;
    
    for (int i = 0; i < layers.size(); i++) {
        layers[i].forward(curr, dest); // applies the activation
        curr = dest;
    }

    assert(dest.size() == nb_classes);
}