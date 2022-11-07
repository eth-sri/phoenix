#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <functional>
#include "utils/ckks_manager.h"

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

using namespace std; 
using namespace seal;

/*
    Counting violations and errors (harmful violations)
*/

struct ViolationData {
    int range_vs = 0;
    int diff_vs = 0;

    int range_errors = 0;
    int diff_errors = 0;

    void add(const ViolationData& other) {
        range_vs += other.range_vs;
        diff_vs += other.diff_vs;
        range_errors += other.range_errors;
        diff_errors += other.diff_errors;
    }

    string print() {
        stringstream ss;
        ss << "Range viols: " << range_vs << " | Diff viols: " << diff_vs;
        ss << " | Range errors: " << range_errors << " | Diff errors: " << diff_errors << endl;
        return ss.str();
    }
};

inline ViolationData count_violations_multiclass(bool VERBOSE, shared_ptr<CKKSManager> ckks, Ciphertext& logits_ctx, int nb_slots, int block_sz, int nb_logits, double D, double L, double R, bool is_prelim) {
    ViolationData data;
    
    vector<double> all_logits;
    ckks->decrypt_and_decode(logits_ctx, all_logits);

    int batch_size = (is_prelim) ? 1 : nb_slots / block_sz;
    for (int it = 0; it < batch_size; it++) {
        // Extract current logits
        auto iter_start = all_logits.begin() + it * block_sz;
        vector<double> logits = {iter_start, iter_start + nb_logits};
        assert(logits.size() == nb_logits);
        
        // Sort decreasing
        sort(logits.begin(), logits.end(), greater<double>());

        // Range violations, all logits (or logit averages if prelim) should be in [0, 1]
        // The only case that actually causes errors is if (max-min)>1!
        for (int i = 0; i < nb_logits; i++) {
            if (logits[i]<0 || logits[i]>1) {
                data.range_vs++;
            }
        }

        double range = logits.front() - logits.back();
        if (range > 1) { 
            if (VERBOSE) if (is_prelim) cout << "(prelim run) ";
            if (VERBOSE) cout << "[ERROR] Violation of logit range at logit array " << it+1 << " (range " << range << ">1) -- this likely breaks everything" << endl;
            data.range_errors++;
        }

        // Diff violations, the diffs must be outside of [-D, D]/2R

        // After first sgn scores are: {-9/17, -7/17, ..., 7/17, 9/17} with error the most ~0.14/17 (accounted for)
        // 8/17 will be the threshold, so 9/17 can't have any mistakes, and the rest should at most reach 7/17 (otw they are safe)
        // 
        // [-D, D] violation with a smaller logit => can reduce the score by some value in [0, 1] => assume 0 and ignore
        // [-D, D] violation with a larger logit => can inc. the score by some value in [0, 1] => assume 1 
        // 7/17 can afford 0 larger-logit-violations, ..., -9/17 can afford 16 larger-logit-violations

        int errors = 0;
        for (int i = 0; i < nb_logits; i++) {
            // Count violations
            int bigger_vs = 0;
            int smaller_vs = 0;
            for (int j = 0; j < nb_logits; j++) {
                if (i == j) continue;
                if (fabs(logits[i] - logits[j]) < D/(R-L)) {
                    if (j < i) bigger_vs++; else smaller_vs++;
                }
            }
            data.diff_vs += bigger_vs + smaller_vs;

            // See if there is an error
            int budget = max(0, (i - 1) * 2);
            int important_vs = (i == 0) ? smaller_vs : bigger_vs;

            if (important_vs > budget) {
                if (VERBOSE) {
                    if (is_prelim) cout << "(prelim run) ";
                    cout << "[ERROR] Too many diff violations at logit array " << it+1 << " and i=" << i;
                    cout << " (" << important_vs << " important violations while budget is " << budget << ")";
                    if (is_prelim) cout << "-- this likely breaks the prelim count" << endl;
                    else cout << " -- this likely causes an off-by-1 in counts" << endl;
                }
                errors++;
            }
        }
        // since only top 2 can err, errors is either 0 or 2
        if (errors == 2) {
            data.diff_errors++;
        } else if (errors != 0) {
            cout << "[ERROR] diff errors should be 0 or 2 but it is" << errors << endl;
        }
        
    }
    return data;
}


inline ViolationData count_violations_binary(bool VERBOSE, shared_ptr<CKKSManager> ckks, Ciphertext& logits_ctx, int nb_slots, int block_sz, int nb_logits, double D, double L, double R, bool is_prelim) {
    ViolationData data;
    
    vector<double> all_logits;
    ckks->decrypt_and_decode(logits_ctx, all_logits);

    int batch_size = (is_prelim) ? 1 : nb_slots / block_sz;
    for (int it = 0; it < batch_size; it++) {
        // Extract current logits
        
        auto iter_start = all_logits.begin() + it * block_sz;
        
        double logit = all_logits[it*block_sz];

        // Range violations, the logits should be in [-1, 1], if not that's bad 
        if (fabs(logit) > 1) {
            data.range_vs++;
            data.range_errors++;
            if (VERBOSE) if (is_prelim) cout << "(prelim run) ";
            if (VERBOSE) cout << "[ERROR] Violation of logit range at logit array " << it+1 << " (logit " << logit << ">1) -- this likely breaks everything" << endl;
        }


        // "Diff violations", the logit must be outside of [-D, D]/R
        if (fabs(logit) < D/R) {
            data.diff_vs++;
            data.diff_errors++;
            if (VERBOSE) {
                if (is_prelim) cout << "(prelim run) ";
                cout << "[ERROR] Diff violation at logit array " << it+1 << "( logit = " << logit << " )";
                if (is_prelim) cout << "-- this likely breaks the prelim count" << endl;
                else cout << " -- this likely causes an off-by-1 in counts" << endl;
            }
        }
    }
    return data;
}