#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <future>
#include "utils/ckks_manager.h"


#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

using namespace std;
using namespace seal;

inline void argmax_multiclass(bool VERBOSE, stringstream& ss_time, shared_ptr<CKKSManager> ckks, Ciphertext& all_logits_ctx, Ciphertext& preds, int nb_slots, int block_sz, int miniblock_sz, int nb_active_miniblocks, int dg1, int df1, int dg2, int df2, int nb_logits) {
    // all_logits_ctx: [block | block | block | ... | block] (batch_size blocks of size block_sz (2048))
    // each block: [miniblock | miniblock | miniblock | 000000] (nb_active_miniblocks (=nb_batches+1) miniblocks of size miniblock_sz (32))
    // [nb_logits-class] each miniblock: [l0, ..., l9, 0, ..., 0], padded with 0s for duplication
    // (!) first miniblock in each block is the preliminary batch (and only first such miniblock is relevant)
    //
    // (output) preds: 1-hot argmax in each relevant miniblock
    // Generate diffs -> Call sgn on each one -> Sum up to get scores -> Second sgn to get one-hot
    auto t_argmax_start = Clock::now();
    if (VERBOSE) cout << endl << "~~~~~~ Starting argmax" << endl;

    // Homomorphically duplicate logits for simple subtraction (to generate diffs)
    Ciphertext tmp;
    ckks->evaluator->rotate_vector(all_logits_ctx, -nb_logits, ckks->galois_keys, tmp);
    ckks->evaluator->add_inplace(all_logits_ctx, tmp);
    auto t_duplication_done = Clock::now();
    if (VERBOSE) log_time(ss_time, "[Smoothing] Duplication of all logits", t_argmax_start, t_duplication_done);


    // 9 times: rotate left, subtract from original, apply sgn
    // sum these up to get a score for each logit: {-9/17, -7/17, ..., 7/17, 9/17}
    int SCALE = 18;
    SgnEvaluator sgn_eval1(VERBOSE, ckks->encoder, 1.0/SCALE);
    Ciphertext all_logits_rotated = all_logits_ctx;

    Ciphertext all_logits_diffs[9];
    Ciphertext partial_scores[9];
    vector<thread> threads;
    for (int i = 0; i <= 8; i++) {
        ckks->evaluator->rotate_vector_inplace(all_logits_rotated, 1, ckks->galois_keys);
        ckks->evaluator->sub(all_logits_ctx, all_logits_rotated, all_logits_diffs[i]);
        // oracle for debug
        if (0 && VERBOSE) {
            vector<double> zzz;
            ckks->decrypt_and_decode(all_logits_diffs[i], zzz); cout << "diffs at " << i << ": ";
            for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            print_minmax(zzz);
        }
        
        // sgn1 in another thread
        auto sgn1_lambda = [](SgnEvaluator& sgn_eval, int dg, int df, Ciphertext& in, Ciphertext& out, shared_ptr<CKKSManager>& ckks) {
            sgn_eval.sgn(dg, df, in, out, ckks);
        };
        thread th(sgn1_lambda, ref(sgn_eval1), dg1, df1, ref(all_logits_diffs[i]), ref(partial_scores[i]), ref(ckks));
        threads.push_back(move(th));
        if (VERBOSE) cout << "Ran thread " << i << endl;

        // ===============
        // NOTE: insta join when doing countops (to disable multithreading)
        // join here but not later
        //threads[i].join(); 
    }

    if (VERBOSE) cout << endl << "~~~~~~ Joining threads" << endl;
    for (int i = 0; i <= 8; i++) {
        if (VERBOSE) cout << "[main] Waiting to join " << i << endl;
        threads[i].join(); 
        if (VERBOSE) cout << "[main] Joined " << i << endl;
    }

    auto t_threads_done = Clock::now();
    if (VERBOSE) log_time(ss_time, "~~~~~~~~~ [Smoothing] Joined all sgn1 threads", t_duplication_done, t_threads_done);

    Ciphertext scores = partial_scores[0];
    for (int i = 1; i <= 8; i++) {
        if (VERBOSE) cout << endl << "~~~ Collecting partial scores " << i << "/" << 8 << endl;
        ckks->evaluator->add_inplace(scores, partial_scores[i]);
        // oracle for debug
        if (0 && VERBOSE) {
            vector<double> zzz;
            ckks->decrypt_and_decode(partial_scores[i], zzz); cout << "partial scores at index " << i << ": ";
            for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            print_minmax(zzz);
        }
        if (0 && VERBOSE) {
            vector<double> zzz;
            ckks->decrypt_and_decode(scores, zzz); cout << "sum of partial scores at index " << i << ": ";
            for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
            print_minmax(zzz);
        }
    }
    
    auto t_scoring_done = Clock::now();
    if (VERBOSE) cout << endl;
    if (VERBOSE) log_time(ss_time, "~~~~~~ [Smoothing] Scoring (first sgn) done", t_duplication_done, t_scoring_done);
    if (VERBOSE) cout << endl;

    // each miniblock starts with nb_logits scores: [s0, ..., s9]
    // si in {-9/17, -7/17, ..., 7/17, 9/17} if bigger than 8/17 this is a win
    // subtract 8/17 to get {-1, ..., -1/17, 1/17}
    // call sign on that to get a 1-hot prediction (rest 0s)
    vector<double> shift(nb_logits, 8.0/SCALE);
    extend_mask_to_small_blocks(shift, nb_slots, block_sz, miniblock_sz, nb_active_miniblocks); // +1 for prelim
    Plaintext shift_plain;
    ckks->encoder->encode(shift, scores.parms_id(), scores.scale(), shift_plain);
    ckks->evaluator->sub_plain_inplace(scores, shift_plain);

    // oracle for debug
    if (0 && VERBOSE) {
        vector<double> zzz;
        ckks->decrypt_and_decode(scores, zzz); cout << "final scores (random 4): ";
        for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        print_minmax(zzz);
    }

    // spend 1 level to mask out irrelevant values before sgn2 (to avoid explosion)
    // scores are ~init_scale but not exactly, after this they're different but this is not an issue
    // (no need to retain init_scale anymore)
    if (VERBOSE) cout << endl << "Masking out irrelevant values before sgn2" << endl << endl;
    vector<double> important_mask(10, 1);
    extend_mask_to_small_blocks(important_mask, nb_slots, block_sz, miniblock_sz, nb_active_miniblocks);
    Plaintext important_mask_plain;
    ckks->encoder->encode(important_mask, scores.parms_id(), scores.scale(), important_mask_plain);
    ckks->evaluator->multiply_plain_inplace(scores, important_mask_plain);
    ckks->evaluator->rescale_to_next_inplace(scores);

    // oracle for debug
    if (0 && VERBOSE) {
        vector<double> zzz;
        ckks->decrypt_and_decode(scores, zzz); cout << "final scores after masking (random 4): ";
        for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        print_minmax(zzz);
    }

    if (VERBOSE) cout << endl << "~~~~~~ Starting second sgn" << endl;
    // {-1/17, 1/17} -> {-0.5, 0.5}
    SgnEvaluator sgn_eval2(VERBOSE, ckks->encoder, 0.5);
    sgn_eval2.sgn(dg2, df2, scores, preds, ckks);

    // add 0.5 to get one-hot argmax, {0, 0, 1, 0, 0}
    vector<double> half(nb_logits, 0.5);
    extend_mask_to_small_blocks(half, nb_slots, block_sz, miniblock_sz, nb_active_miniblocks); // +1 for prelim
    Plaintext half_plain;
    ckks->encoder->encode(half, preds.parms_id(), preds.scale(), half_plain);
    ckks->evaluator->add_plain_inplace(preds, half_plain);

    // oracle for debug
    if (0 && VERBOSE) {
        vector<double> zz;
        ckks->decrypt_and_decode(preds, zz); cout << "final preds (random 4): ";
        for (int i = 0; i < nb_logits; i++) cout << zz[i] << " "; cout << endl;
        for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zz[i] << " "; cout << endl;
        for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zz[i] << " "; cout << endl;
        for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zz[i] << " "; cout << endl;
        print_minmax(zz);
    }

    auto t_argmax_done = Clock::now();
    if (VERBOSE) log_time(ss_time, "~~~~~~ [Smoothing] Getting preds (second sgn) done", t_scoring_done, t_argmax_done);
}


inline void argmax_binary(bool VERBOSE, stringstream& ss_time, shared_ptr<CKKSManager> ckks, Ciphertext& all_logits_ctx, Ciphertext& preds, int nb_slots, int block_sz, int miniblock_sz, int nb_active_miniblocks, int dg, int df) {
    // all_logits_ctx: [block | block | block | ... | block] (batch_size blocks of size block_sz (256))
    // each block: [miniblock | miniblock | miniblock | 000000] (nb_batches+1 miniblocks of size miniblock_sz (2))
    // [binary] each miniblock: [-l, l]
    // (!) first miniblock in each block is the preliminary batch (and only first such miniblock is relevant)
    //
    // (output) preds: 1-hot argmax in each relevant miniblock
    // Contrary to multiclass here the logits -l/l are already the scores in [-1, 1]
    // We can just do the scores->preds part directly, and some postprocessing to get 1-hot

    auto t_argmax_start = Clock::now();
    if (VERBOSE) cout << endl << "Starting argmax" << endl;

    // scores in [-1, 1] -> flags: {-0.5, 0.5}
    SgnEvaluator sgn_eval(VERBOSE, ckks->encoder, 0.5);
    sgn_eval.sgn(dg, df, all_logits_ctx, preds, ckks);

    // {+-0.5} -> {0,1}
    vector<double> half = {0.5, 0.5};
    extend_mask_to_small_blocks(half, nb_slots, block_sz, miniblock_sz, nb_active_miniblocks); // +1 for prelim
    Plaintext half_plain;
    ckks->encoder->encode(half, preds.parms_id(), preds.scale(), half_plain);
    ckks->evaluator->add_plain_inplace(preds, half_plain);

    // oracle for debug
    if (VERBOSE) {
        int nb_logits = 2;
        vector<double> zzz;
        ckks->decrypt_and_decode(preds, zzz); cout << "final preds (random 4): ";
        for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        print_minmax(zzz);
    }

    auto t_argmax_done = Clock::now();
    if (VERBOSE) log_time(ss_time, "[Smoothing] Argmax_binary done", t_argmax_start, t_argmax_done);
}