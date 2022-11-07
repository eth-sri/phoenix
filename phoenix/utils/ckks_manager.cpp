#include <vector>
#include <iomanip>
#include <iostream>

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

#include "utils/ckks_manager.h"

using namespace std;
using namespace seal;

uint64_t CKKSManager::get_modulus(Ciphertext& x, int k) {
    // Returns k-th (from 1) prime in the chain of x 
    const vector<Modulus>& modulus = context->get_context_data(x.parms_id())->parms().coeff_modulus();
    int sz = modulus.size();
    return modulus[sz-k].value();
}

void CKKSManager::encode_and_encrypt(const vector<double>& msg, Ciphertext& dest) {
    Plaintext msg_plain;
    encoder->encode(msg, init_scale, msg_plain);
    encryptor->encrypt(msg_plain, dest);
}

void CKKSManager::decrypt_and_decode(const Ciphertext& ctx, vector<double>& dest) {
    Plaintext ctx_plain;
    decryptor->decrypt(ctx, ctx_plain);
    encoder->decode(ctx_plain, dest);
}


void CKKSManager::init_ckks(bool VERBOSE) {
    // Prepare the modulus switching chain: 60 [50 ... 50] special=60
    int nb_bits = log2(init_scale);
    if (VERBOSE) cout << "nb_bits: " << nb_bits << endl;
    if (VERBOSE) cout << "Chose N=" << N << " so max bit count is " << CoeffModulus::MaxBitCount(N) << endl;
    if (VERBOSE) cout << "Will attempt to use " << 60 + nb_bits*depth + 60 << " bits (depth = " << depth << ")" << endl;

    vector<int> moduli_bits(depth+2, nb_bits);
    moduli_bits[0] = 60; 
    moduli_bits[moduli_bits.size() - 1] = 60;

    // Set up context
    // use ::tc128 for 128-bit security (supported up to 1<<15)
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);

    auto moduli = CoeffModulus::Create(N, moduli_bits);
    parms.set_coeff_modulus(moduli);
    
    context = make_shared<SEALContext>(parms, true, sec_level_type::none);

    // Generate keys
    KeyGenerator keygen(*context);
    keygen.create_public_key(public_key);
    secret_key = keygen.secret_key();
    keygen.create_relin_keys(relin_keys);

    // Generate galois keys
    vector<int> rots;
    for (int i = 1; i <= N/4; i *= 2) {
        rots.insert(rots.end(), {-i, i});
    }
    keygen.create_galois_keys(rots, galois_keys);

    // Set up tools
    encoder = make_shared<CKKSEncoder>(*context);
    encryptor = make_shared<Encryptor>(*context, public_key);
    decryptor = make_shared<Decryptor>(*context, secret_key);
    evaluator = make_shared<Evaluator>(*context);
}