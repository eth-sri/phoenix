#include "linalg.h"

using namespace std;

vector<double> add(vector<double> a, vector<double> b) {
  if (a.size()!=b.size()) {
    throw invalid_argument("vectors must have the same dimensions.");
  } else {
    vector<double> c(a.size());
    for (int i = 0; i < a.size(); i++) {
      c[i] = a[i] + b[i];
    }
    return c;
  }
}

vector<double> mult(vector<double> a, vector<double> b) {
  if (a.size()!=b.size()) {
    throw invalid_argument("vectors must have the same dimensions.");
  } else {
    vector<double> c(a.size());
    for (int i = 0; i < a.size(); i++) {
      c[i] = a[i]*b[i];
    }
    return c;
  }
}

vector<double> diag(vector<vector<double>> M, int d) {
  const int m = M.size();
  const int n = m > 0 ? M[0].size() : 0;
  if (m==0 || n==0 || m > n) {
    throw invalid_argument("matrix must have non-zero dimensions and must have m <= n.");
  }
  if (d > n) {
    throw invalid_argument("Invalid Diagonal Index.");
  }
  vector<double> diag(n);
  for (int k = 0; k < n; k++) {
    diag[k] = M[k%m][(k + d)%n];
  }
  return diag;
}

vector<double> get_random_vector(int dim) {
  vector<double> v(dim);
  for (int j = 0; j < dim; j++) {
    v[j] = (static_cast<double>(rand())/RAND_MAX) - 0.5;
  }
  return v;
}

vector<vector<double>> diagonals_from_matrix(const vector<vector<double>> M) {
  const int m = M.size();
  const int n = m > 0 ? M[0].size() : 0;
  if (m==0 || n==0 || m > n) {
    throw invalid_argument("matrix must have non-zero dimensions and must have m <= n.");
  }
  vector<vector<double>> diagonals(m);
  for (int i = 0; i < M.size(); ++i) {
    diagonals[i] = diag(M, i);
  }
  return diagonals;
}

vector<double> duplicate(const vector<double> v) {
  int dim = v.size();
  vector<double> r;
  r.reserve(2*dim);
  r.insert(r.begin(), v.begin(), v.end());
  r.insert(r.end(), v.begin(), v.end());
  return r;
}

vector<double> mvp_hybrid_ptx(vector<vector<double>> diagonals, vector<double> v) {
  const int m = diagonals.size();
  if (m==0) {
    throw invalid_argument(
        "matrix must not be empty!");
  }
  const int n = diagonals[0].size();
  if (n!=v.size() || n==0) {
    throw invalid_argument(
        "matrix and vector must have matching non-zero dimension");
  }
  int n_div_m = n/m;
  int log2_n_div_m = ceil(log2(n_div_m));
  if (m*n_div_m!=n || (2ULL << (log2_n_div_m - 1)!=n_div_m && n_div_m!=1)) {
    throw invalid_argument(
        "matrix dimension m must divide n and the result must be power of two");
  }

  // Hybrid algorithm based on "GAZELLE: A Low Latency Framework for Secure Neural Network Inference" by Juvekar et al.
  // Available at https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar
  // Actual Implementation based on the description in
  // "DArL: Dynamic Parameter Adjustment for LWE-based Secure Inference" by Bian et al. 2019.
  // Available at https://ieeexplore.ieee.org/document/8715110/ (paywall)

  vector<double> t(n, 0);
  for (int i = 0; i < m; ++i) {
    vector<double> rotated_v = v;
    rotate(rotated_v.begin(), rotated_v.begin() + i, rotated_v.end());
    auto temp = mult(diagonals[i], rotated_v);
    t = add(t, temp);
  }

  vector<double> r = t;
  //TODO: if n/m isn't a power of two, we need to masking/padding here
  for (int i = 0; i < log2_n_div_m; ++i) {
    vector<double> rotated_r = r;
    int offset = n/(2ULL << i);
    rotate(rotated_r.begin(), rotated_r.begin() + offset, rotated_r.end());
    r = add(r, rotated_r);
  }

  r.resize(m);

  return r;
}


void mvp_hybrid_ctx(const seal::GaloisKeys &galois_keys, seal::Evaluator &evaluator,
                                            seal::CKKSEncoder &encoder, int m, int n,
                                            vector<vector<double>> diagonals,
                                            const seal::Ciphertext &ctv, seal::Ciphertext &enc_result, uint64_t p) {
  if (m==0 || m!=diagonals.size()) {
    throw invalid_argument(
        "Matrix must not be empty, and diagonals vector must have size m!");
  }
  
  /*if (n!=diagonals[0].size() || n==0) {
    throw invalid_argument(
        "Diagonals must have non-zero dimension that matches n");
  }*/
  
  int n_div_m = n/m;
  int log2_n_div_m = ceil(log2(n_div_m));
  if (m*n_div_m!=n || (2ULL << (log2_n_div_m - 1)!=n_div_m && n_div_m!=1)) {
    throw invalid_argument(
        "Matrix dimension m must divide n and the result must be power of two");
  }

  // Hybrid algorithm based on "GAZELLE: A Low Latency Framework for Secure Neural Network Inference" by Juvekar et al.
  // Available at https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar
  // Actual Implementation based on the description in
  // "DArL: Dynamic Parameter Adjustment for LWE-based Secure Inference" by Bian et al. 2019.
  // Available at https://ieeexplore.ieee.org/document/8715110/ (paywall)

  //  vec t(n, 0);
  seal::Ciphertext ctxt_t;

  for (int i = 0; i < m; ++i) {

    // rotated_v = rot(v,i)
    seal::Ciphertext ctxt_rotated_v = ctv;
    if ( i != 0)  evaluator.rotate_vector_inplace(ctxt_rotated_v, i, galois_keys);

    // auto tmp = mult(diagonals[i], rotated_v);
    seal::Plaintext ptxt_current_diagonal;
    encoder.encode(diagonals[i], ctxt_rotated_v.parms_id(), p, ptxt_current_diagonal);
    seal::Ciphertext ctxt_tmp;
    evaluator.multiply_plain(ctxt_rotated_v, ptxt_current_diagonal, ctxt_tmp);

    // t = add(t, tmp);
    if (i==0) {
      ctxt_t = ctxt_tmp;
    } else {
      evaluator.add_inplace(ctxt_t, ctxt_tmp);
    }
  }

  // vec r = t;
  seal::Ciphertext ctxt_r = move(ctxt_t);

  //TODO: if n/m isn't a power of two, we need to masking/padding here
  for (int i = 0; i < log2_n_div_m; ++i) {
    // vec rotated_r = r;
    seal::Ciphertext ctxt_rotated_r = ctxt_r;

    // Calculate offset
    int offset = n/(2ULL << i);

    // rotated_r = rot(rotated_r, offset)
    evaluator.rotate_vector_inplace(ctxt_rotated_r, offset, galois_keys);

    // r = add(r, rotated_r);
    evaluator.add_inplace(ctxt_r, ctxt_rotated_r);
  }
  //  r.resize(m); <- has to be done by the client
  // for efficiency we do not mask away the other entries
  enc_result = move(ctxt_r);
}
