#pragma once

#include <vector>
#include <memory>
#include "utils/ckks_manager.h"
#include "smoothing/stats.h"

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

// Corrupts both the vector<double> image and the Ciphertext image_ctx with stddev of sigma
inline void corrupt(std::mt19937& engine, std::normal_distribution<>& gaussian, std::shared_ptr<CKKSManager> ckks, std::vector<double>& image, std::vector<double>& cimage, seal::Ciphertext& image_ctx, seal::Ciphertext& cimage_ctx) {
    int input_size = image.size();

    // Generate noise
    std::vector<double> noise(input_size, 0);
    for (int i = 0; i < input_size; i++) {
        noise[i] = gaussian(engine);
    }

    // Add noise homomorphically (we need to duplicate it, as we duplicated the image as well)
    seal::Plaintext noise_plain;
    ckks->encoder->encode(duplicate(noise), image_ctx.parms_id(), image_ctx.scale(), noise_plain);
    ckks->evaluator->add_plain(image_ctx, noise_plain, cimage_ctx);

    // Add noise in plain
    std::transform(image.begin(), image.end(), noise.begin(), cimage.begin(), std::plus<double>());
}

// Corrupts both the vector<double> clean_batch and the Ciphertext clean_batch_ctx with stddev of sigma
// Assuming duplicated images of input size input_size
// TODO merge duplicated code below
inline void corrupt_batch(std::mt19937& engine, std::normal_distribution<>& gaussian, std::shared_ptr<CKKSManager> ckks, std::vector<double>& clean_batch, std::vector<double>& batch, seal::Ciphertext& clean_batch_ctx, seal::Ciphertext& batch_ctx, int batch_size, int input_size) {
    // Generate noise
    std::vector<double> noise(clean_batch.size(), 0);

    for (int i = 0; i < batch_size; i++) {
        int idx_start = 2*input_size*i;
        for (int j = 0; j < input_size; j++) {
            int idx = idx_start+j;
            noise[idx+input_size] = noise[idx] = gaussian(engine);
        }
    }

    // Add noise homomorphically
    seal::Plaintext noise_plain;
    ckks->encoder->encode(noise, clean_batch_ctx.parms_id(), clean_batch_ctx.scale(), noise_plain);
    ckks->evaluator->add_plain(clean_batch_ctx, noise_plain, batch_ctx);

    // Add noise in plain
    std::transform(clean_batch.begin(), clean_batch.end(), noise.begin(), batch.begin(), std::plus<double>());
}

// Corrupts both the vector<double> clean_batch and the Ciphertext clean_batch_ctx with stddev of sigma
// Assuming duplicated images of input size input_size
inline void corrupt_batch_uniform(std::mt19937& engine, std::uniform_real_distribution<>& uniform, std::shared_ptr<CKKSManager> ckks, std::vector<double>& clean_batch, std::vector<double>& batch, seal::Ciphertext& clean_batch_ctx, seal::Ciphertext& batch_ctx, int batch_size, int input_size) {
    // Generate noise
    std::vector<double> noise(clean_batch.size(), 0);

    for (int i = 0; i < batch_size; i++) {
        int idx_start = 2*input_size*i;
        for (int j = 0; j < input_size; j++) {
            int idx = idx_start+j;
            noise[idx+input_size] = noise[idx] = uniform(engine);
        }
    }

    // Add noise homomorphically
    seal::Plaintext noise_plain;
    ckks->encoder->encode(noise, clean_batch_ctx.parms_id(), clean_batch_ctx.scale(), noise_plain);
    ckks->evaluator->add_plain(clean_batch_ctx, noise_plain, batch_ctx);

    // Add noise in plain
    std::transform(clean_batch.begin(), clean_batch.end(), noise.begin(), batch.begin(), std::plus<double>());
}


// Same as corrupt_batch but with multivariate
inline void corrupt_batch(std::mt19937& engine, std::vector<std::normal_distribution<>>& gaussians, std::shared_ptr<CKKSManager> ckks, std::vector<double>& clean_batch, std::vector<double>& batch, seal::Ciphertext& clean_batch_ctx, seal::Ciphertext& batch_ctx, int batch_size, int input_size) {
    assert (gaussians.size() == input_size);

    // Generate noise
    std::vector<double> noise(clean_batch.size(), 0);

    for (int i = 0; i < batch_size; i++) {
        int idx_start = 2*input_size*i;
        for (int j = 0; j < input_size; j++) {
            int idx = idx_start+j;
            noise[idx+input_size] = noise[idx] = gaussians[j](engine);
        }
    }

    // Add noise homomorphically
    seal::Plaintext noise_plain;
    ckks->encoder->encode(noise, clean_batch_ctx.parms_id(), clean_batch_ctx.scale(), noise_plain);
    ckks->evaluator->add_plain(clean_batch_ctx, noise_plain, batch_ctx);

    // Add noise in plain
    std::transform(clean_batch.begin(), clean_batch.end(), noise.begin(), batch.begin(), std::plus<double>());
}
