#include <cassert>
#include <iomanip>

#include "learnable_square.h"

using namespace std;
using namespace seal;

LearnableSquare::LearnableSquare(shared_ptr<CKKSManager> ckks, vector<double> a, vector<double> b) {
    this->ckks = ckks;
    this->a = a;
    this->b = b;
}


// Homomorphic inference
// ax^2+bx = (ax+b)x
void LearnableSquare::forward(Ciphertext& input, Ciphertext& dest) {
    double D = input.scale();
    uint64_t p = ckks->get_modulus(input, 1);
    uint64_t q = ckks->get_modulus(input, 2);
    double a_scale = (double)p/D * q;
    // scale(a) := pq/D
    // => scale(ax) = scale(ax+b) = q 
    // => scale((ax+b) * x) = D

    // ax
    Plaintext a_plain;
    ckks->encoder->encode(a, input.parms_id(), a_scale, a_plain);
    Ciphertext tmp;
    ckks->evaluator->multiply_plain(input, a_plain, tmp);
    ckks->evaluator->rescale_to_next_inplace(tmp);

    // ax+b
    Plaintext b_plain;
    ckks->encoder->encode(b, tmp.parms_id(), tmp.scale(), b_plain);
    ckks->evaluator->add_plain_inplace(tmp, b_plain);

    // (ax+b)x
    ckks->evaluator->mod_switch_to_next_inplace(input);
    ckks->evaluator->multiply(tmp, input, dest);
    ckks->evaluator->relinearize_inplace(dest, ckks->relin_keys);
    ckks->evaluator->rescale_to_next_inplace(dest);

    assert(fabs(dest.scale() - D) < 1e-3); // sanity check
}

// Standard inference
// ax^2+bx = (ax+b)x
void LearnableSquare::forward(vector<double>& input, vector<double>& dest) {
    vector<double> tmp(input.size());
    transform(input.begin(), input.end(), a.begin(), tmp.begin(), multiplies<double>());
    transform(tmp.begin(), tmp.end(), b.begin(), tmp.begin(), plus<double>());
    transform(tmp.begin(), tmp.end(), input.begin(), dest.begin(), multiplies<double>());
}