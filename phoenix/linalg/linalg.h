#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

/**
 * \brief Addition between two vectors (component-wise). Both vectors must have the same length
 * \param a vector of any length d
 * \param b vector of same length d
 * \return The sum between a and b, a vector of the same length d as the inputs
 * \throw std::invalid_argument if the dimensions mismatch
 */
std::vector<double> add(std::vector<double> a, std::vector<double> b);

/**
 * \brief Multiplication between two vectors (component-wise). Both vectors must have the same length
 * \param a vector of any length d
 * \param b vector of same length d
 * \return The component-wise product between a and b, a vector of the same length d as the inputs
 * \throw std::invalid_argument if the dimensions mismatch
 */
std::vector<double> mult(std::vector<double> a, std::vector<double> b);

/**
 * \brief The d-th (generalized) diagonal of a matrix. The matrix M must be "squat".
 * \param M A matrix of size m x n, where m <= n
 * \param d Index of the diagonal, where d = 0 is the main diagonal. Wraps around, i.e. d = n is the last diagonal (the one below main diagonal)
 * \return d-th diagonal  of M, a vector of length m
 * \throw std::invalid_argument if M is non-squat or d is geq than n
 */
std::vector<double> diag(std::vector<std::vector<double>> M, int d);

/**
 * \brief Generates a vector with random values from [-1/2,1/2]
 * \param dim Length of the vector
 * \return A vector of length dim with values in [-1/2,1/2]
 */
std::vector<double> get_random_vector(int dim);

/**
 * \brief Returns a list of all the (generalized) diagonals of a "squat" matrix. Numbering starts with the main diagonal and moves up with wrap-around, i.e. the last element is the diagonal one below the main diagonal).
 * \param M A matrix of size m x n, where m <= n
 * \return The list of length m of all the diagonals of M, each a vector of length n
 * \throw std::invalid_argument if M is non-squat
 */
std::vector<std::vector<double>> diagonals_from_matrix(const std::vector<std::vector<double>> M);

/**
 * \brief Returns a vector of twice the length, with the elements repeated in the same sequence
 * \param v vector of length d
 * \return vector of length 2*d that contains two concatenated copies of the input vector
 */
std::vector<double> duplicate(const std::vector<double> v);

/**
 * \brief Computes the matrix-vector-product between a matrix M represented by its diagonals, and a vector.
 *  Plaintext implementation of the FHE-optimized approach due to Juvekar et al. (Hybrid diagonal-representation) and the baby-step giant-step algorithm
 * \param diagonals matrix of size m x n represented by the its (generalized) diagonals (numbering starts with the main diagonal and moves up with wrap-around, i.e. the last element is the diagonal one below the main diagonal)
 * \param v vector of length n
 * \return The matrix-vector product between M and v, a vector of length n
 * \throw std::invalid_argument if the dimensions mismatch
 */
std::vector<double> mvp_hybrid_ptx(std::vector<std::vector<double>> diagonals, std::vector<double> v);


/**
 * \brief Compute the matrix-vector-product between a squat plaintext matrix, represented by its diagonals, and an encrypted vector.
 *  Uses the hybrid algorithm based on "GAZELLE: A Low Latency Framework for Secure Neural Network Inference" by Juvekar et al.
 *  *ATTENTION*: Batching must be done in a way so that if the matrix has dimension m x n, rotating the vector left n times results in a correct cyclic rotation of the first n elements!
 *  This is usually done by simply duplicating the vector, e.g. using function duplicate(vec x), if the number of slots in the ciphertexts and the dimension of the vector are not the same
 * \param[in] galois_keys Rotation keys, should allow arbitrary rotations (reality is slightly more complicated due to baby-step--giant-step algorithm)
 * \param[in] evaluator Evaluation object from SEAL
 * \param[in] encoder Encoder object from SEAL
 * \param[in] m First dimension of matrix. Must divide n and ** m/n must be a power of two **
 * \param[in] n Length of the vector, which must match the second dimension of the matrix
 * \param[in] diagonals The plaintext matrix, represented by the its diagonals (numbering starts with the main diagonal and moves up with wrap-around, i.e. the last element is the diagonal one below the main diagonal)
 * \param[in] ctv The encrypted vector, batched into a single ciphertext. The length must match n
 * \param[out] enc_result  Encrypted vector, batched into a single ciphertext
 * \param[in] p Scale to use for diags
 * \throw std::invalid_argument if the dimensions mismatch or m/n is not a power of two.
 */
void mvp_hybrid_ctx(const seal::GaloisKeys& galois_keys, seal::Evaluator& evaluator,
                                         seal::CKKSEncoder& encoder, int m, int n,
                                         std::vector<std::vector<double>> diagonals,
                                         const seal::Ciphertext& ctv, seal::Ciphertext& enc_result, uint64_t p);