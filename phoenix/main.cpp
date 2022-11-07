#include "hcephes.h"
  
#include <vector>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <future>
#include <thread>
#include <ctime>  

#include "utils/jsoncpp/json-forwards.h"
#include "utils/jsoncpp/json.h"
#include "utils/rapidcsv.h"
#include "utils/ckks_manager.h"
#include "utils/helpers.h"
#include "linalg/linalg.h"
#include "nn/mlp.h"
#include "nn/dataloader.h"
#include "smoothing/corrupt.h"
#include "smoothing/sgn.h"
#include "smoothing/stats.h"
#include "smoothing/violations.h"
#include "smoothing/argmax.h"

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

using namespace std;
using namespace seal; 

int main(int argc, char *argv[]) {
    #ifdef MOCK
        cout << "This is a mock run, using mock_seal.h" << endl << endl;
    #else
        cout << "This is a real run, using Microsoft SEAL 3.6.6" << endl << endl;
    #endif
    assert(argc==3); // only the path of the config file and example idx

    //////////////////////////////////////////////////////////////////////////////////
    //                               PARAMS                                         //
    //////////////////////////////////////////////////////////////////////////////////
    Json::Value args = load_config(argv[1]);
    bool VERBOSE     = args["VERBOSE"].asBool();              // Everything will be printed
    
    if (VERBOSE) cout << "args loaded from " << argv[1] << ":\n" << args << endl;

    // >>>>>>>>>>>>>>>>>>>> Meta params
    int seed         = args["seed"].asInt();                  // Set -1 for random seed
    srand(seed);
    string out_path  = args["out_path"].asString();           // Where to save the outputs

    // >>>>>>>>>>>>>>>>>>>> Dataset params
    string dataset   = args["dataset"].asString();            
    bool is_binary   = (dataset == string("adult"));          // If this is a binary or a multiclass problem
    string data_path = args["data_path"].asString();          // Path to .csv with data
    int nb_examples  = args["nb_examples"].asInt();           // How many examples are in the csv file
    int input_size   = args["input_size"].asInt();            // Input size padded to pow2
    int nb_logits    = args["nb_logits"].asInt();             // Actual nb logits (not padded to pow2)

    bool rand_data = false;
    if (input_size < 0) {
        // Negative input sizes mean random data
        rand_data = true;
        input_size *= -1;
    }

    // >>>>>>>>>>>>>>>>>>>> Network params
    string weights_path    = args["weights_path"].asString();   // Path to .csv with model weights
    vector<int> net_layers = parse_int_vector(args["layers"]);  // Sizes of dense layers in the net

    // >>>>>>>>>>>>>>>>>>>> Smoothing params
    int example_idx = args["example_idx"].asInt();              // On which example to evaluate
    example_idx = atoi(argv[2]);                                // Command line override

    double sigma; vector<double> sigmas; // Noise intensity
    if (is_binary) {
        sigmas = parse_double_vector(args["sigmas"]);
    } else {
        sigma = args["sigma"].asDouble();
    }
    
    int Ns0         = args["Ns0"].asInt();                    // Number of samples for prelim count
    int Ns          = args["Ns"].asInt();                     // Number of samples for main count
    double tau      = args["tau"].asDouble();                 // Fixed level we try to prove
    double alpha    = args["alpha"].asDouble();               // Error probability

    // >>>>>>>>>>>>>>>>>>>> Argmax params
    double L = args["L"].asDouble();                                // Logits must be >= L
    double R = args["R"].asDouble();                                // Logits must be <= R
    double D = args["D"].asDouble();                                // Logit diffs must be >= D
    if (is_binary) assert(L == -R); // positive must stay positive
    double net_divide = (is_binary) ? R  : (R-L);        // Divide weights & bias by this
    double net_add    = (is_binary) ? 0 : 1 - R/(R-L);   // Add this to bias

    int dg1 = args["dg1"].asInt(); // Argmax 1st polynomial (dg)
    int df1 = args["df1"].asInt(); // Argmax 1st polynomial (df), set to 0 for binary
    int dg2 = args["dg2"].asInt(); // Argmax 2nd polynomial (dg), set to 0 for binary
    int df2 = args["df2"].asInt(); // Argmax 2nd polynomial (df)

    // >>>>>>>>>>>>>>>>>>>> CKKS params
    int N = (1 << args["log_N"].asInt());                    // poly_modulus_degree
    int nb_slots = (N >> 1);                                 // N/2 slots
    double init_scale = pow(2.0, args["log_delta"].asInt()); // delta
    int block_sz = 2*input_size;                             // space between examples in a ciphertext/batch (2048 / 256)
    int miniblock_sz = (!is_binary) ? 32 : 2;                // space between two logit spaces in all_logits_ctx (32 / 2)
    int supported_nb_batches = block_sz / miniblock_sz;      // (64 / 128) *batches* of logits can fit in one ctx
    std::cout << "Supported nb batches:" << supported_nb_batches << std::endl;
    
    // >>>>>>>>>>>>>>>>>>>> Batch and depth calculations
    int batch_size = nb_slots / block_sz;      // how many inputs can we fit in one batch (256 / 32)
    int nb_batches = Ns / batch_size;          // nb_batches ignoring the preliminary batch

    int depth_nn = 4*net_layers.size()-3;      // 1 for linear, 2 for learnable act, 1 before linear for rot
    int depth_reduce = 1;
    int depth_argmax = 4*(dg1+df1+dg2+df2) +((is_binary)?0:1); // potentially add one to mask out between sgns
    int depth_mask_winner = 1;
    int depth = depth_nn + depth_reduce + depth_argmax + depth_mask_winner;
    if (VERBOSE) cout << "Using depth=" << depth << endl;

    // >>>>>>>>>>>>>>>>>>>> Asserts
    int last = input_size;
    for (int x : net_layers) {
        // FastMVP requirement: input = output * 2^k
        assert(last%x == 0);
        assert(ispow2(last/x));
        last = x;
    }
    assert(example_idx < nb_examples);                       // need to pick a valid example
    assert(Ns0 == batch_size);                               // do exactly one prelim batch 
    assert(Ns % batch_size == 0);                            // all real batches shuld be full
    assert(ispow2(nb_batches));                              // for simplicity of counting
    assert(ispow2(batch_size));                              // for simplicity of counting
    assert(supported_nb_batches >= (1+nb_batches)); // preliminary will be there too so it should fit

    // Dump all args for debug
    if (0) {
        cout << VERBOSE << " " << out_path << " " << seed << endl;
        cout << dataset << " " << is_binary << " " << data_path << " " << nb_examples << " " << input_size << " " << nb_logits << endl;
        cout << weights_path << " "; for (int x : net_layers) cout << x << " "; cout << endl; 
        cout << example_idx << " " << sigma << " " << Ns0 << " " << Ns << " " << tau << " " << alpha << endl;
        cout << L << " " << R << " " << D << " " << net_divide << " " << net_add << " " << dg1 << " " << df1 << " " << dg2 << " " << df2 << endl;
        cout << N << " " << nb_slots << " " << init_scale << " " << block_sz << " " << miniblock_sz << endl;
        cout << batch_size << " " << nb_batches << " " << depth_nn << " " << depth_reduce << " " << depth_argmax << " " << depth_mask_winner << " " << depth << endl;
    }

    //////////////////////////////////////////////////////////////////////////////////
    //                       SET UP: INPUTS / CKKS CONTEXT                          //
    //////////////////////////////////////////////////////////////////////////////////

    // Log times
    stringstream ss_time;
    ss_time << "[TIMES]\n";
    auto t_start = Clock::now();
    time_t tt_start = chrono::system_clock::to_time_t(t_start);
    if (VERBOSE) cout << "Current timestamp: " << ctime(&tt_start) << endl;

    // Set up CKKSManager
    shared_ptr<CKKSManager> ckks = make_shared<CKKSManager>(N, depth, init_scale, VERBOSE);
    auto t_after_setup = Clock::now();
    if (VERBOSE) log_time(ss_time, "[C/S] Set up CKKS", t_start, t_after_setup);
    time_t tt_ckks = chrono::system_clock::to_time_t(t_after_setup);
    if (VERBOSE) cout << "Current timestamp: " << ctime(&tt_ckks) << endl;

    // Load the input and encrypt it
    Dataloader dataloader(dataset, data_path, input_size, rand_data);
    vector<double> image;
    bool div = !is_binary;
    dataloader.get_input(example_idx, image, div);
    int target = dataloader.get_target(example_idx);
    if (VERBOSE) cout << "Testing on example #" << example_idx << " (target = " << target << ")" << endl;
    Ciphertext image_ctx;
    ckks->encode_and_encrypt(image, image_ctx); // User sends [image000...000]
    auto t_data_loaded = Clock::now();
    if (VERBOSE) log_time(ss_time, "[C] Load input", t_after_setup, t_data_loaded);

    //////////////////////////////////////////////////////////////////////////////////
    //                       SERVER: INFERENCE FOR SMOOTHING                        //
    //////////////////////////////////////////////////////////////////////////////////

    // mlp for the main count (outputs in [0,1], for binary:[-1,1])
    MLP mlp(VERBOSE, ckks, block_sz, nb_slots, input_size, 
        net_layers, weights_path, true, net_divide, net_add);

    // mlp0 for the prelim count (outputs in [0, 1/Ns0], for binary:[-1/Ns0, 1/Ns0], so sum gives average)
    MLP mlp0(VERBOSE, ckks, block_sz, nb_slots, input_size, 
        net_layers, weights_path, true, net_divide*Ns0, net_add/Ns0);
    
    auto t_mlp = Clock::now();
    if (VERBOSE) log_time(ss_time, "[S1] Params and MLP initialized", t_data_loaded, t_mlp);
    // /* opCheckpoint */ckks->evaluator->opCheckpoint(); // init, just to reset 
  
    // Repeat the input image 2*batch_size times to create a clean batch (that will be corrupted in each prop)
    assert(input_size*2*batch_size == nb_slots);
    int log2_repetitions = log2(2*batch_size);
    Ciphertext clean_batch_ctx = image_ctx;
    Ciphertext tmp;
    for (int i = 0; i < log2_repetitions; i++) {
        int offset = input_size * (1 << i);
        ckks->evaluator->rotate_vector(clean_batch_ctx, -offset, ckks->galois_keys, tmp);
        ckks->evaluator->add_inplace(clean_batch_ctx, tmp);
    }

    // [Do the same in the clear for testing]
    vector<double> clean_batch;
    repeat(image, clean_batch, 2*batch_size);
    assert(clean_batch.size() == nb_slots);

    // Put all logits in one ciphertext
    Ciphertext all_logits_ctx;

    // Logit masks that will be needed for all_logits_ctx
    vector<double> logit_mask((is_binary) ? 1 : nb_logits, 1); // for binary leave 0 at other spot
    extend_mask_to_blocks(logit_mask, nb_slots, block_sz);
    vector<double> neg_logit_mask((is_binary) ? 1 : nb_logits, -1); // for binary
    extend_mask_to_blocks(neg_logit_mask, nb_slots, block_sz);

    // Counts: calculate in plaintext to check
    vector<double> expected_prelim_counts(nb_logits, 0);
    vector<double> expected_counts(nb_logits, 0);

    // Violations
    ViolationData vdata_prelim;
    ViolationData vdata;

    // Prepare randomness, if seed==-1 seed with random device
    random_device rd{};
    mt19937 engine{(seed == -1) ? rd() : seed};
    normal_distribution<> gaussian;
    uniform_real_distribution<> uniform;
    vector<normal_distribution<>> gaussians;
    bool use_uniform = false;
    if (is_binary) {
        // TODO unify
        for (double sigma : sigmas) {
            gaussians.push_back(normal_distribution<>(0, sigma));
        }
        while (gaussians.size() < input_size) {
            gaussians.push_back(normal_distribution<>(0, 0));
        }
    } else {
        if (sigma < 0) { // Negative noise means uniform
            sigma *= -1;
            use_uniform = true;
            cout << "Using uniform noise" << endl;
            uniform = uniform_real_distribution<double>(-sigma, sigma);
        } else {
            gaussian = normal_distribution<>(0, sigma);
        }
    }

    if (VERBOSE) cout << "--------------------> Starting propagation of " << nb_batches << " batches with batch size " << batch_size << " + 1 prelim batch" << endl;
    auto t_propagation_start = Clock::now();

    // Corrupted images batch (in cipher and in clear for testing)
    vector<Ciphertext> batch_ctxs(nb_batches+1);
    vector<vector<double>> batches(nb_batches+1);

    // Output logits
    vector<Ciphertext> logits_ctxs(nb_batches+1);

    // Start forward prop
    vector<thread> threads; // This will always be <=cores so no need for threadpool
    auto t_prop_start = Clock::now();
    if (VERBOSE) log_time(ss_time, "[S2] [Smoothing] Repetition++", t_mlp, t_prop_start);
    // /* opCheckpoint SKIP */ckks->evaluator->opCheckpoint(); // Input duplication
    if (VERBOSE) cout << endl << "--------------------> Adding noise and running threads" << endl;

    // Batch corrupt
    for (int b_it = 0; b_it <= nb_batches; b_it++) { // last is the preliminary one
        // Fill the ciphertext with corrupted inputs
        batches[b_it].resize(nb_slots, 0);
        if (is_binary) {
            corrupt_batch(engine, gaussians, ckks, clean_batch, batches[b_it], clean_batch_ctx, batch_ctxs[b_it], batch_size, input_size);   
        } else {
            if (use_uniform) {
                cout << "Taking also noise from uniform" << endl;
                // TODO unify
                corrupt_batch_uniform(engine, uniform, ckks, clean_batch, batches[b_it], clean_batch_ctx, batch_ctxs[b_it], batch_size, input_size);
            } else {
                corrupt_batch(engine, gaussian, ckks, clean_batch, batches[b_it], clean_batch_ctx, batch_ctxs[b_it], batch_size, input_size);
            }
        }
    }

    // /* opCheckpoint */double COST1 = ckks->evaluator->opCheckpoint();

    // Run threads
    for (int b_it = 0; b_it <= nb_batches; b_it++) { // last is the preliminary one
        double is_prelim = (b_it == nb_batches);
        MLP& curr_mlp = (is_prelim) ? mlp0 : mlp;

        // Inference in another thread
        auto prop_lambda = [](MLP& curr_mlp, Ciphertext& in, Ciphertext& out) {
            curr_mlp.forward(in, out);
        };

        // NOTE: disable multithreading when doing countops, join here but not later
        thread th(prop_lambda, ref(curr_mlp), ref(batch_ctxs[b_it]), ref(logits_ctxs[b_it]));
        threads.push_back(move(th));
        if (VERBOSE) cout << "Ran thread " << b_it << endl;
        //threads[b_it].join(); 

    }

    auto t_corrupted = Clock::now();
    if (VERBOSE) log_time(ss_time, "[S3] [Smoothing] Batch corruption + running threads", t_prop_start, t_corrupted);
    
    if (VERBOSE) cout << endl << "--------------------> Joining threads" << endl;
    for (int b_it = 0; b_it <= nb_batches; b_it++) { 
        if (VERBOSE) cout << "[main] Waiting to join " << b_it << endl;
        threads[b_it].join(); 
        if (VERBOSE) cout << "[main] Joined " << b_it << endl;
    } // Join all

    auto t_joined = Clock::now();
    if (VERBOSE) log_time(ss_time, "[S4] [Smoothing] Joined all threads", t_corrupted, t_joined);
    // /* opCheckpoint skip */ckks->evaluator->opCheckpoint(); // Inference

    // Finish properly in order
    for (int b_it = 0; b_it <= nb_batches; b_it++) { // last is the preliminary one
        if (VERBOSE) cout << endl << "--------------------> Collecting batch " << b_it+1 << "/" << nb_batches+1 << endl;

        double is_prelim = (b_it == nb_batches);

        MLP& curr_mlp = (is_prelim) ? mlp0 : mlp;
        Ciphertext& logits_ctx = logits_ctxs[b_it];
        vector<double>& batch = batches[b_it];

        // Plain inference + check
        // Comment out when measuring latency
        
        vector<vector<double>> expected_logits;
        for (int i = 0; i < batch_size; i++) {
            expected_logits.push_back({});
            vector<double> tmp(input_size, 0);
            for (int j = 0; j < input_size; j++) {
                tmp[j] = batch[i*(2*input_size) + j];
            }
            curr_mlp.forward(tmp, expected_logits[i]);
        }
        vector<double> logits;
        ckks->decrypt_and_decode(logits_ctx, logits);

        int to_print; 
        #ifdef MOCK 
            to_print = (is_prelim) ? 2 : 2; 
        #else 
            to_print = batch_size;
        #endif
        if (VERBOSE) cout << "Homomorphic inference:" << endl;
        if (VERBOSE) {print_logits_batched(logits, net_layers.back(), nb_slots, block_sz, target, to_print); print_minmax(logits);}
        if (VERBOSE) cout << "Plain inference:" << endl;
        if (VERBOSE) print_logits_batched(to_print, expected_logits, net_layers.back(), target);

        // Plain counting to test
        if (VERBOSE) cout << "Plain counting" << endl;
        for (auto& exp_logits : expected_logits) {
            if (is_binary) exp_logits = {-exp_logits.back(), exp_logits.back()};
            int winner = argmax(exp_logits, nb_logits);
            if (is_prelim) {
                expected_prelim_counts[winner]++;
            } else {
                expected_counts[winner]++;
            }
        }
        

        Ciphertext logits_ctx_masked;
        // Mask the logits so that all else are 0s (need to generate all_logits_ctx)
        // logits_ctx should be ~init_scale, we preserve this
        uint64_t p = ckks->get_modulus(logits_ctx, 1);
        Plaintext logit_mask_plain;
        ckks->encoder->encode(logit_mask, logits_ctx.parms_id(), p, logit_mask_plain);
        ckks->evaluator->multiply_plain(logits_ctx, logit_mask_plain, logits_ctx_masked);
        ckks->evaluator->rescale_to_next_inplace(logits_ctx_masked);

        // If binary also mask the logits with -1 and place in the other slot
        // So we get two logits: [l, -l]
        // Does not waste levels
        if (is_binary) {
            Ciphertext neg_logits_ctx_masked;
            Plaintext neg_logit_mask_plain;
            ckks->encoder->encode(neg_logit_mask, logits_ctx.parms_id(), p, neg_logit_mask_plain);
            ckks->evaluator->multiply_plain(logits_ctx, neg_logit_mask_plain, neg_logits_ctx_masked);
            ckks->evaluator->rescale_to_next_inplace(neg_logits_ctx_masked);
            ckks->evaluator->rotate_vector_inplace(logits_ctx_masked, -1, ckks->galois_keys); // [0, l]
            ckks->evaluator->add_inplace(logits_ctx_masked, neg_logits_ctx_masked); // [-l ,l]
        }

        // If this is the prelim run: sum all logits/Ns0 into the first nb_logits slots (average)
        // This gets added as usual, the rest of this batch in all_logits_ctx is therefore #-values
        if (is_prelim) {
            Ciphertext rot;
            // (first: all blocks -> first block)
            int log2_batch_size = log2(batch_size);
            for (int i = 0; i < log2_batch_size; i++) {
                int offset = nb_slots / (1 << (i+1));
                ckks->evaluator->rotate_vector(logits_ctx_masked, offset, ckks->galois_keys, rot);
                ckks->evaluator->add_inplace(logits_ctx_masked, rot);
            }
        }

        // Count violations (Comment out when measuring latency)
        ViolationData vs;
        if (is_binary) {
            vs = count_violations_binary(VERBOSE, ckks, logits_ctx_masked, nb_slots, block_sz, nb_logits, D, L, R, is_prelim);
        } else {
            vs = count_violations_multiclass(VERBOSE, ckks, logits_ctx_masked, nb_slots, block_sz, nb_logits, D, L, R, is_prelim);
        }
        if (is_prelim) vdata_prelim = vs; else vdata.add(vs);
        if (VERBOSE) {
            cout << "Violations at batch end:";
            cout << (is_prelim) ? vdata_prelim.print() : vdata.print();
            cout << endl;
        }
        
        // Add the new entry
        if (b_it == 0) {
            all_logits_ctx = logits_ctx_masked;
        } else {
            // rotate old by miniblock_size and insert the new entry
            ckks->evaluator->rotate_vector_inplace(all_logits_ctx, -miniblock_sz, ckks->galois_keys);
            ckks->evaluator->add_inplace(all_logits_ctx, logits_ctx_masked);
        }
    }
    auto t_all_inference_done = Clock::now();
    if (VERBOSE) cout << endl;
    if (VERBOSE) log_time(ss_time, "--------------------> [S5] [Smoothing] Collected all - all inference done! Total time", t_joined, t_all_inference_done);
    ///* opCheckpoint */double COST2 = ckks->evaluator->opCheckpoint(); // Reduction
    time_t tt_inf = chrono::system_clock::to_time_t(t_all_inference_done);
    if (VERBOSE) cout << "Current timestamp: " << ctime(&tt_inf) << endl;
    if (VERBOSE) cout << "Expected counts:" << endl;
    if (VERBOSE) for (int i = 0; i < nb_logits; i++) cout << expected_counts[i] << " "; 
    if (VERBOSE) cout << endl;

    // Oracle for debug
    if (0 && VERBOSE) {
        vector<double> zzz;
        ckks->decrypt_and_decode(all_logits_ctx, zzz); cout << "all_logits_ctx in the end (some 4): " << endl;
        for (int i = 0; i < nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = miniblock_sz; i < miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+0; i < block_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
        for (int i = block_sz+miniblock_sz; i < block_sz+miniblock_sz+nb_logits; i++) cout << zzz[i] << " "; cout << endl;
    }


    //////////////////////////////////////////////////////////////////////////////////
    //                       SERVER: ARGMAX                                         //
    //////////////////////////////////////////////////////////////////////////////////

    // All inference is done now 
    // all_logits_ctx: [block | block | block | ... | block] (batch_size blocks of size block_sz (2048 / 256))
    // each block: [miniblock | miniblock | miniblock | 000000] (nb_batches+1 miniblocks of size miniblock_sz (32 / 2))
    // [10-class] each miniblock: [l0, ..., l9, 0, ..., 0], padded with 0s for duplication
    // [binary] each miniblock: [-l, l]
    // (!) first miniblock in each block is the preliminary batch (and only first such miniblock is relevant, rest is #-values)
    //
    // Output of argmax = preds: 1-hot argmax in each relevant miniblock (for both 10-class and binary)

    // all_logits_ctx is ~init_scale but not exactly
    if (VERBOSE) cout << endl << " --------------------> Argmax" << endl;
    Ciphertext preds;
    if (is_binary) {
        argmax_binary(VERBOSE, ss_time, ckks, all_logits_ctx, preds, nb_slots, block_sz, miniblock_sz, nb_batches+1, dg1, df2);
    } else {
        argmax_multiclass(VERBOSE, ss_time, ckks, all_logits_ctx, preds, nb_slots, block_sz, miniblock_sz, nb_batches+1, dg1, df1, dg2, df2, nb_logits);
    }
    // each g4/f4 application will try to keep the scale the same (up to float error) 
    // but we don't stabilize anything from now until the end as there's no real need to do so, any scales work for final masking

    auto t_argmax_done = Clock::now();
    if (VERBOSE) cout << endl;
    if (VERBOSE) log_time(ss_time, "--------------------> [S6] Argmax", t_all_inference_done, t_argmax_done);
    ///* opCheckpoint */double COST3 = ckks->evaluator->opCheckpoint(); // Argmax

    //////////////////////////////////////////////////////////////////////////////////
    //                       SERVER: COUNTING                                       //
    //////////////////////////////////////////////////////////////////////////////////
    if (VERBOSE) cout << endl << " --------------------> Counting" << endl;
    auto t_count_start = Clock::now();

    // time to split preliminary count (winner mask) and the rest
    Ciphertext winner_mask = preds; // first nb_logits are the mask, rest is garbage

    // rotate preds left to kick out the preliminary count, we are back at nb_batches, like before
    ckks->evaluator->rotate_vector_inplace(preds, miniblock_sz, ckks->galois_keys);

    // rotations to sum up all one-hot winners 
    Ciphertext rot;
    // first: all blocks -> first block
    int log2_batch_size = log2(batch_size);
    for (int i = 0; i < log2_batch_size; i++) {
        int offset = nb_slots / (1 << (i+1));
        ckks->evaluator->rotate_vector(preds, offset, ckks->galois_keys, rot);
        ckks->evaluator->add_inplace(preds, rot);
    }
    // second: all miniblocks -> first miniblock
    int log2_nb_batches = log2(nb_batches);
    for (int i = 0; i < log2_nb_batches; i++) {
        int offset = (miniblock_sz*nb_batches) / (1 << (i+1));
        ckks->evaluator->rotate_vector(preds, offset, ckks->galois_keys, rot);
        ckks->evaluator->add_inplace(preds, rot);
    }
    
    // Now first nb_logits are the final counts
    Ciphertext counts_ctx = preds;

    auto t_count_end = Clock::now();
    if (VERBOSE) log_time(ss_time, "[S7] [Smoothing] Counting done", t_argmax_done, t_count_end);
   // /* opCheckpoint skip */ckks->evaluator->opCheckpoint(); // Aggregation

    if (VERBOSE) cout << endl << "------------> [Server results]" << endl << endl;

    // Oracle for debug
    vector<double> final_winner_mask;
    ckks->decrypt_and_decode(winner_mask, final_winner_mask); if (VERBOSE) cout << "winner mask: " << endl;
    for (int i = 0; i < nb_logits; i++) if (VERBOSE) cout << final_winner_mask[i] << " "; if (VERBOSE) cout << endl;

    // Assert: winner mask valid, we don't check the winner as this is checked later from returned values
    // mask_zero > 1/2Ns => masked could be > 1/2
    // mask_one < 1 - 1/2Ns => masked could have a mistake
    int nb_ones = 0;
    int winner_mask_winner;
    bool winner_mask_valid = true;
    for (int i = 0; i < nb_logits; i++) {
        if (final_winner_mask[i] < 0.5) {
            if (fabs(final_winner_mask[i]) > 1.0/(2*Ns)) {
                if (VERBOSE) cout << "[server ERROR] winner mask has too big 0-values" << endl;
                winner_mask_valid = false;
            }
        } else {
            if (fabs(final_winner_mask[i] - 1) > 1.0/(2*Ns)) {
                if (VERBOSE) cout << "[server ERROR] winner mask has too small 1-values" << endl;
                winner_mask_valid = false;
            }
            nb_ones++;
            winner_mask_winner = i;
        }
    }
    if (nb_ones == 0) {
        if (VERBOSE) cout << "[server ERROR] winner mask has no 1-values" << endl;
        winner_mask_valid = false;
    }
    if (nb_ones > 1) {
        if (VERBOSE) cout << "[server ERROR] winner mask has more 1-values" << endl;
        winner_mask_valid = false;
    }
    if (VERBOSE) cout << "[server] Winner mask valid: " << winner_mask_valid << " with nb_ones=" << nb_ones << " and winner=" << winner_mask_winner << endl;

    // Oracle for debug
    vector<double> final_counts;
    ckks->decrypt_and_decode(counts_ctx, final_counts); if (VERBOSE) cout << "counts: " << endl;
    for (int i = 0; i < nb_logits; i++) if (VERBOSE) cout << final_counts[i] << " "; if (VERBOSE) cout << endl;
    if (VERBOSE) cout << "[server] counts (rounded): " << endl;
    for (int i = 0; i < nb_logits; i++) {
        final_counts[i] = round(final_counts[i]); if (VERBOSE) cout << final_counts[i] << " ";
    }
    if (VERBOSE) cout << endl;


    // outputs: 
    //          winner_mask = [0,.,1,.,0] / [0, 1]
    //          counts_ctx  = [c0,...,c9] / [c0, c1]
    

    //////////////////////////////////////////////////////////////////////////////////
    //                       SERVER: BINTEST                                        //
    //////////////////////////////////////////////////////////////////////////////////

    if (VERBOSE) cout << endl << "------------>  Final step: bintest" << endl;

    // Subtract k from the counts, but add one 
    // (so F>=1 is the winning condition instead of F>=0 as F=0 is undetectable)
    int k = get_min_n_winner_to_not_abstain(Ns, tau, alpha);
    vector<double> k_vec(nb_logits, (double)(k-1));
    Plaintext k_plain;
    ckks->encoder->encode(k_vec, counts_ctx.parms_id(), counts_ctx.scale(), k_plain);
    ckks->evaluator->sub_plain_inplace(counts_ctx, k_plain);

    // Mask out the winner
    
    Ciphertext flag_hot_ctx;
    if (VERBOSE) cout << "Masking out, scales are: " << setprecision(30) << counts_ctx.scale() << " and " << winner_mask.scale() << " (both should be around" << init_scale << ")" << setprecision(6) << endl;
    ckks->evaluator->multiply(counts_ctx, winner_mask, flag_hot_ctx);
    ckks->evaluator->rescale_to_next_inplace(flag_hot_ctx); 
    // note: maybe not exactly init scale

    // Get the radius
    double radius = (is_binary) ? 0 : get_radius(sigma, tau);

    // return value (flag_hot_ctx):
    //  [0, 0, ..., F, ... 0]
    // 1. round everything
    // 2. if (F>=1) we predict idx(F) robustly with radius R
    //    else we abstain
    auto t_all_end = Clock::now();
    if (VERBOSE) log_time(ss_time, "[S8] [Smoothing] Bintest done", t_count_end, t_all_end);
    ///* opCheckpoint */double COST4 = ckks->evaluator->opCheckpoint(); // Bintest 
    if (VERBOSE) log_time(ss_time, "[Smoothing] All done! Total time", t_start, t_all_end);
    if (VERBOSE) log_time(ss_time, "[S->FULL] Total time on server (without setup)", t_data_loaded, t_all_end);

    if (VERBOSE) cout << "Returning [flag_hot_ctx] and [R] to the client. Server done." << endl;


    //////////////////////////////////////////////////////////////////////////////////
    //                       CLIENT: DECRYPT/DECODE/EVAL                            //
    //////////////////////////////////////////////////////////////////////////////////

    if (VERBOSE) cout << endl << endl << "------------> [Back on the client]" << endl << endl;

    // Expected counts
    if (VERBOSE) cout << "[[[Expectation]]] (doing everything in the clear)" << endl;
    if (VERBOSE) cout << "Expected preliminary counts:" << endl;
    for (int i = 0; i < nb_logits; i++) if (VERBOSE) cout << expected_prelim_counts[i] << " "; 
    if (VERBOSE) cout << endl;
    int expected_winner = argmax(expected_prelim_counts, nb_logits);
    if (VERBOSE) cout << "=> Expected winner: " << expected_winner << endl;
    if (VERBOSE) cout << "Expected counts:" << endl;
    for (int i = 0; i < nb_logits; i++) if (VERBOSE) cout << expected_counts[i] << " "; 
    if (VERBOSE) cout << endl;

    // Expected bintest
    if (VERBOSE) cout << "Expected bintest:" << endl;
    int expected_k = get_min_n_winner_to_not_abstain(Ns, tau, alpha);
    double expected_radius = (is_binary) ? 0 : get_radius(sigma, tau);
    if(fabs(expected_radius-radius) > 1e-9) cout << "[ERROR] Radius from the server is different (" << expected_radius << " vs " << radius << endl;

    int expected_n_winner = expected_counts[expected_winner];
    bool expected_test_passed = (expected_n_winner >= expected_k);
    if (VERBOSE) {
        cout << "    Params: " << "Ns=" << Ns << ", tau=" << tau << ", alpha=" << alpha << ", sigma=";
        if (is_binary) {
            for (double s : sigmas) cout << s << " "; cout << endl;
        } else {
            cout << sigma << endl;
        }
    }
    if (VERBOSE) cout << "    Needed counts to not abstain: " << expected_k << endl;
    if (VERBOSE) cout << "    Comparing winner counts: " << expected_n_winner << " of class " << expected_winner << " (diff is " << expected_n_winner-expected_k+1 << ")" << endl;
    if (expected_test_passed) {
        if (is_binary) {
            if (VERBOSE) cout << "    Expectation: Test passed, we predict " << expected_winner << " certifiably fair!" << endl;
        } else {
            if (VERBOSE) cout << "    Expectation: Test passed, we robustly predict " << expected_winner << " with radius " << expected_radius << endl;
        }

    } else {
        if (VERBOSE) cout << "    Expectation: ABSTAIN (need " << expected_k-expected_n_winner << " more)" << endl;
    }
    if (VERBOSE) cout << "Reminder: target class is " << target << " (ignored)" << endl;

    // Decrypt the flag hot and extract the result 
    // (used only for final correctness as we could get all 0s in which case we don't know the winner but we also don't care)
    if (VERBOSE) cout << endl << endl << "[[[Actual]]] (decrypting the HE results)" << endl;
    vector<double> flag_hot;
    ckks->decrypt_and_decode(flag_hot_ctx, flag_hot);
    if (VERBOSE) cout << "Flag hot:" << endl;
    for (int i = 0; i < nb_logits; i++) if (VERBOSE) cout << flag_hot[i] << " "; 
    if (VERBOSE) cout << endl;
    if (VERBOSE) cout << "Flag hot (rounded):" << endl;
    int flag=0, winner=-1, nonzeros=0;
    for (int i = 0; i < nb_logits; i++) {
        flag_hot[i] = round(flag_hot[i]);
        if (VERBOSE) cout << flag_hot[i] << " ";
        if (flag_hot[i] != 0) {
            nonzeros++;
            flag = flag_hot[i];
            winner = i;
        }
    }
    if (VERBOSE) cout << endl;

    // Verdict
    if (nonzeros > 1) {
        if (VERBOSE) cout << "[ERROR] There should be at most one non-zero element in the flag-hot vector" << endl;
        // (1) There could be all 0s if the winner count is (k-1)
    } else {
        if (flag <= 0) { // Also if all zeros
            if (VERBOSE) cout << "Flag is negative! We ABSTAIN";
        } else {
            if (is_binary) {
                if (VERBOSE) cout << "Flag is positive! We predict class " << winner << " while being certifiably fair (had " << flag-1 << " more counts than needed)" << endl;
            } else {
                if (VERBOSE) cout << "Flag is positive! We robustly predict class " << winner << " with radius " << radius << " (had " << flag-1 << " more counts than needed)" << endl;
            }
        }
    }
    if (VERBOSE) cout << "--> Client done." << endl;


    //////////////////////////////////////////////////////////////////////////////////
    //                       CHECK FOR CORRECTNESS                                  //
    //////////////////////////////////////////////////////////////////////////////////

    if (VERBOSE) cout << endl << endl << "------------> [Checking for correctness]" << endl << endl;

    // Based on the winner mask (prelim count) -> not sent to the client
    // Q: Was the heuristic good enough to select the right winner? Server-side 1
    bool winner_correct = winner_mask_valid && (winner_mask_winner == expected_winner);
    
    // Based on the final counts -> not sent to the client
    // Q: Were the counts we had exactly correct? Server-side 2
    bool counts_correct = true; // if so, assert that flag is also correct
    for (int i = 0; i < nb_logits; i++) { 
        if (expected_counts[i] != final_counts[i]) counts_correct = false;
    }
    if (counts_correct && flag != (expected_n_winner-expected_k+1)) cout << "[ERROR] Counts correct but flag wrong" << endl;

    // Winner mask + counts = decision

    // Q: Is the result client got even valid according to API? Client-side 1
    bool result_valid = (nonzeros <= 1); 

    // Q: Is the response client got correct? Client-side 2
    bool result_correct = false;
    result_correct |= (flag <= 0 && !expected_test_passed); // abstained as we should have
    result_correct |= (flag > 0 && expected_test_passed) && (winner == expected_winner); // winner correct and predicted
    result_correct &= result_valid; // if the result is invalid it doesn't matter

    // End
    auto t_end = Clock::now();
    log_time(ss_time, "Total time (without setup)", t_after_setup, t_end);
    log_time(ss_time, "Total time (with setup)", t_start, t_end);
    time_t tt_endd = chrono::system_clock::to_time_t(t_end);
    if (VERBOSE) cout << "Current timestamp: " << ctime(&tt_endd) << endl;

    // Correctness
    if (VERBOSE) {
        cout << "Example idx = " << example_idx << endl;
        cout << "~~~~~~~~~~ CORRECTNESS of HE:" << endl;
        cout << "    [?] original model is correct on this example: " << (expected_winner == target) << endl;
        cout << "    [?] original model is certified on this example: " << (expected_test_passed) << endl;
        cout << "    [C] result_valid  : " << result_valid << endl;
        cout << "    [S] winner_correct: " << winner_correct << endl;
        cout << "    [S] counts_correct: " << counts_correct << endl;
        cout << "    [C] result_correct: " << result_correct << endl;
        cout << endl;

        cout << "Violation Data          ||| ";
        cout << vdata.print();
        cout << "Violation Data (Prelim) ||| "; 
        cout << vdata_prelim.print();
    }

    // Save results to file
    stringstream line;
    line << "example," << int(example_idx);
    
    line << ",ORIG";
    line << "," << int(expected_winner == target);
    line << "," << int(expected_test_passed);

    line << ",US";
    line << "," << int(result_valid);
    line << "," << int(winner_correct); // can be wrong but if we abstain it doesn't matter
    line << "," << int(counts_correct);
    line << "," << int(result_correct); // OK

    line << ",ERRORS";
    line << "," << int(vdata.range_vs > 0);
    line << "," << int(vdata.range_errors > 0);
    line << "," << int(vdata.diff_vs > 0);
    line << "," << int(vdata.diff_errors > 0);
    line << "," << int(vdata_prelim.range_vs > 0);
    line << "," << int(vdata_prelim.range_errors > 0);
    line << "," << int(vdata_prelim.diff_vs > 0);
    line << "," << int(vdata_prelim.diff_errors > 0);
    line << endl;

    if (VERBOSE) cout << "Appending this line to file " << out_path << ":" << endl;
    cout << line.str();

    ofstream out;
    out.open(out_path, ios_base::app);
    out << line.str();

    // Microbenchmarking report
    /*
    double L1 = std::chrono::duration_cast<std::chrono::milliseconds>(t_corrupted - t_mlp).count();
    double L2 = std::chrono::duration_cast<std::chrono::milliseconds>(t_all_inference_done - t_corrupted).count();
    double L3 = std::chrono::duration_cast<std::chrono::milliseconds>(t_argmax_done - t_all_inference_done).count();
    double L4 = std::chrono::duration_cast<std::chrono::milliseconds>(t_all_end - t_argmax_done).count();
    double TOTAL = std::chrono::duration_cast<std::chrono::milliseconds>(t_all_end - t_mlp).count();
    cout << "Microbenchmark report [ms]: " << L1 << " " << L2 << " " << L3 << " " << L4 << " | TOTAL is " << TOTAL << endl;
    */

    // CountOPS summary Uncomment along with opCheckpoints to count ops 
    /*
    cout << "Cost report [(+p)s]: " << COST1 << " " << COST2 << " " << COST3 << " " << COST4 << " | TOTAL is hopefully " << COST1+COST2+COST3+COST4 << endl;

    ckks->evaluator->opCheckpoint();

    cout << "Cost report SCALED: " << fixed << setprecision(1) << 1.0 << " " << COST2/COST1 << " " << COST3/COST1 << " " << COST4/COST1 << endl;
    */

    return 0;
}