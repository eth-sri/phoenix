#pragma once

/*
	header-only mock implementation of the parts of the SEAL library used in the main code,
	can be swapped instead of real SEAL with no other changes at call site 
	
	will be used if the precompiler flag "MOCK" is set (`make mockphoenix` instead of `make phoenix`)

	performs some asserts (modulus chains, scales, levels), but does not take into account FHE noise => useful for debugging circuit designs (and decoupling circuit errors from errors due to FHE noise)
*/

#include <cstddef>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <algorithm>
#include <cassert>

using namespace std;

namespace seal
{

enum class sec_level_type : int { none = 0x0, some = 0x1 };
enum class scheme_type : std::uint8_t { bfv = 0x1, ckks = 0x2 };

class Modulus {
public:
	Modulus(uint64_t val) {this->val = val;}
	uint64_t value() const {return val;}
	uint64_t val;
};

class CoeffModulus {
public:
	static constexpr int MaxBitCount(int N) {
		switch(N) {
			case 1024: return 27;
			case 2048: return 54;
			case 4096: return 109;
			case 8192: return 218;
			case 16384: return 438;
			case 32768: return 881;
		}
		return 0;
	}
	static vector<Modulus> Create(int N, vector<int>& bits) {
		// we just keep the bits
		vector<Modulus> moduli;
		for (auto& b : bits) {
			moduli.push_back(Modulus(b));
		}
		return moduli;
	}
};

class EncryptionParameters {
public:
	EncryptionParameters() {};
	EncryptionParameters(scheme_type st) {};
	EncryptionParameters(int N, vector<Modulus> moduli) {this->N = N; this->moduli = moduli;}
	void set_poly_modulus_degree(int N) {this->N = N;}
	void set_coeff_modulus(vector<Modulus> moduli) {this->moduli = moduli; this->moduli.pop_back();}
	vector<Modulus> coeff_modulus() {return moduli;}

	int N;
	vector<Modulus> moduli;
};

class ContextData {
public:
	ContextData(EncryptionParameters parms) {this->data = parms;}
	EncryptionParameters parms() {return data;}
	EncryptionParameters data;
};

class SEALContext {
public:
	SEALContext() {};
	SEALContext(EncryptionParameters parms, bool dummy1, sec_level_type dummy2) {
		this->parms = parms;
	}
	ContextData* get_context_data(EncryptionParameters some_parms) {
		ContextData* ctx_data = new ContextData(some_parms);
		return ctx_data;
	}
	EncryptionParameters parms;
};

class Key {
public:

};
typedef Key PublicKey;
typedef Key SecretKey;
typedef Key RelinKeys;
typedef Key GaloisKeys;

class KeyGenerator {
public:
	KeyGenerator(SEALContext& context) {
		this->context = context;
	}	
	void create_public_key(PublicKey k) {};
	SecretKey secret_key() {SecretKey k; return k;}
	void create_relin_keys(RelinKeys k) {};
	void create_galois_keys(vector<int> rot_steps, GaloisKeys k) {};

	SEALContext context;
};

typedef pair<int, vector<int>> parms;

class Text {
public:
	Text() {};
	int coeff_modulus_size() const {return moduli.size();}
	int poly_modulus_degree() const {return nb_slots*2;}
	[[nodiscard]] inline auto &scale() noexcept {scale_ = 1; return scale_;}
	EncryptionParameters parms_id() {return EncryptionParameters(nb_slots*2, moduli);}


	int nb_slots;
	vector<Modulus> moduli;
	double scale_ = 1.0;

	vector<double> m;
};
typedef Text Ciphertext;
typedef Text Plaintext;

class CKKSEncoder {
public:
	CKKSEncoder(SEALContext context) {
		this->moduli = context.parms.moduli;
		nb_slots = context.parms.N / 2;
	}
	void encode(const vector<double>& msg, EncryptionParameters parms, double scale, Plaintext& msg_plain) {
		msg_plain.nb_slots = parms.N / 2;
		msg_plain.moduli = parms.moduli;
		msg_plain.scale() = 1.0; 

		msg_plain.m = vector<double>(msg);
		msg_plain.m.resize(msg_plain.nb_slots);
	}


	void encode(const vector<double>& msg, double scale, Plaintext& msg_plain) {
		EncryptionParameters parms({nb_slots*2, moduli});
		encode(msg, parms, scale, msg_plain);
	}

	void encode(double val, double scale, Plaintext& msg_plain) {
		vector<double> msg(nb_slots, val);
		encode(msg, scale, msg_plain);
	}

	void encode(double val, EncryptionParameters parms, double scale, Plaintext& msg_plain) {
		vector<double> msg(nb_slots, val);
		encode(msg, parms, scale, msg_plain);
	}
	
	void decode(Plaintext& msg_plain, vector<double>& dest) {
		dest = msg_plain.m;
	}

	vector<Modulus> moduli;
	int nb_slots;
};

class Encryptor {
public:
	Encryptor(SEALContext context, PublicKey k) {
		//this->moduli = context.parms.moduli;
		//nb_slots = context.parms.N / 2;
	}
	void encrypt(Plaintext msg_plain, Ciphertext& msg_ctx) {
		msg_ctx = msg_plain;
	}

	//vector<int> moduli;
	//int nb_slots;
};

class Decryptor {
public:
	Decryptor(SEALContext context, SecretKey k) {
		//this->moduli = context.parms.moduli;
		//nb_slots = context.parms.N / 2;
	}
	void decrypt(Ciphertext msg_ctx, Plaintext& msg_plain) {
		msg_plain = msg_ctx;
	}

	//vector<int> moduli;
	//int nb_slots;
};

// NOTE: scale accounting doesn't work

class Evaluator {
public:

	int loc[8] = {0,0,0,0,0,0,0,0};
	int glob[8] = {0,0,0,0,0,0,0,0};
	double weights[8] = {1.53, 1.00, 4.73, 2.58, 80.7, 18.99, 81.26, 467.57}; // for benchmarking

	double opCheckpoint() {
		double locS = 0;
		std::cout << "[CntOp] [Local] ";
		for (int i = 0; i < 8; i++) {
			locS += loc[i] * weights[i];
			std::cout << loc[i] << " ";
			if (i == 3) std::cout << "|";

			loc[i] = 0;
		}
		std::cout << "[" << locS << "]" << std::endl;

		double globS = 0;
		std::cout << "[CntOp] [Global] ";
		for (int i = 0; i < 8; i++) {
			globS += glob[i] * weights[i];
			std::cout << glob[i] << " ";
			if (i == 3) std::cout << "|";
		}
		std::cout << "[" << globS << "]" << std::endl;
		return locS; // return score
	}

	Evaluator(SEALContext context) {
		//this->moduli = context.parms.moduli;
		//nb_slots = context.parms.N / 2;
		std::cout << "[CntOp] " << "Initialized evaluator" << std::endl;

	}

	// Add
	void add(Text& ctx1, Text& ctx2, Text& dest, bool direct=true) {
		if (direct) {
			loc[0]++; glob[0]++;
		}
		assert(ctx1.nb_slots == ctx2.nb_slots); // slots
		assert(ctx1.moduli.size() == ctx2.moduli.size()); // levels
		assert(fabs(ctx1.scale() - ctx2.scale()) < 1e-6); // scale

		dest.nb_slots = ctx1.nb_slots;
		dest.moduli = ctx1.moduli;
		dest.scale() = ctx1.scale();

		dest.m.resize(dest.nb_slots);
		transform(ctx1.m.begin(), ctx1.m.end(), ctx2.m.begin(), dest.m.begin(), std::plus<double>());
	}
	void add_plain(Ciphertext& ctx, Plaintext& ptx, Ciphertext& dest) {
		add(ctx, ptx, dest, false);
		loc[1]++; glob[1]++;
	}
	void add_inplace(Ciphertext& ctx, Ciphertext& ctx_other) {
		add(ctx, ctx_other, ctx, false);
		loc[0]++; glob[0]++;
	}
	void add_plain_inplace(Ciphertext& ctx, Plaintext& ptx_other) {
		add(ctx, ptx_other, ctx, false);
		loc[1]++; glob[1]++;
	}

	// Sub
	void sub(Text& ctx1, Text& ctx2, Text& dest, bool direct=true) {
		if (direct) {
			loc[0]++; glob[0]++;
		}
		assert(ctx1.nb_slots == ctx2.nb_slots); // slots
		assert(ctx1.moduli.size() == ctx2.moduli.size()); // levels
		assert(fabs(ctx1.scale() - ctx2.scale()) < 1e-6); // scale

		dest.nb_slots = ctx1.nb_slots;
		dest.moduli = ctx1.moduli;
		dest.scale() = ctx1.scale();

		dest.m.resize(dest.nb_slots);
		transform(ctx1.m.begin(), ctx1.m.end(), ctx2.m.begin(), dest.m.begin(), std::minus<double>());
	}
	void sub_inplace(Ciphertext& ctx, Ciphertext& ctx_other) {
		sub(ctx, ctx_other, ctx, false);
		loc[1]++; glob[1]++;
	}
	void sub_plain_inplace(Ciphertext& ctx, Plaintext& ptx_other) {
		sub(ctx, ptx_other, ctx, false);
		loc[1]++; glob[1]++;
	}

	// Mul
	void multiply(Text& ctx1, Text& ctx2, Text& dest, bool direct=true) {
		if (direct) {
			loc[2]++; glob[2]++;
		}
		assert(ctx1.nb_slots == ctx2.nb_slots); // slots
		assert(ctx1.moduli.size() == ctx2.moduli.size()); // levels

		dest.nb_slots = ctx1.nb_slots;
		dest.moduli = ctx1.moduli;
		//dest.scale() = ctx1.scale() + ctx2.scale();

		dest.m.resize(dest.nb_slots);
		transform(ctx1.m.begin(), ctx1.m.end(), ctx2.m.begin(), dest.m.begin(), std::multiplies<double>());
	}
	void square(Ciphertext& ctx, Ciphertext& dest) {
		multiply(ctx, ctx, dest, false);
		loc[2]++; glob[2]++;
	}
	void square_inplace(Ciphertext& ctx) {
		multiply(ctx, ctx, ctx, false);
		loc[2]++; glob[2]++;
	}
	void multiply_plain(Ciphertext& ctx, Plaintext& plain, Ciphertext& dest) {
		multiply(ctx, plain, dest, false);
		loc[3]++; glob[3]++;
	}
	void multiply_inplace(Ciphertext& ctx, Ciphertext& other) {
		multiply(ctx, other, ctx, false);
		loc[2]++; glob[2]++;
	}
	void multiply_plain_inplace(Ciphertext& ctx, Plaintext& plain) {
		multiply(ctx, plain, ctx, false);
		loc[3]++; glob[3]++;
	}

	// Rot

	inline bool ispow2(int x) {
		x = abs(x);
		while (x % 2 == 0) x /= 2;
		return (x == 1);
	}

	void rotate_vector_inplace(Ciphertext& ctx, int steps, const GaloisKeys& k) {

		// Negative right, positive left
		auto middle = (steps > 0) ? ctx.m.begin()+steps : ctx.m.begin()+ctx.m.size()-abs(steps);
		rotate(ctx.m.begin(), middle, ctx.m.end());
		if (ispow2(steps)) {
			loc[6]++; glob[6]++;
		} else {
			loc[7]++; glob[7]++;
		}
	}
	void rotate_vector(Ciphertext ctx, int steps, const GaloisKeys& k, Ciphertext& dest) {
		dest = ctx;
		rotate_vector_inplace(dest, steps, k);
	}

	// Rescale
	void rescale_to_next_inplace(Text& text) {
		//text.scale_ -= text.moduli.back().value();
		text.moduli.pop_back();
		assert (text.moduli.size() > 0); // the last prime can't be popped
		loc[5]++; glob[5]++;
	}

	// Modswitch
	void mod_switch_to_next_inplace(Text& text) {
		text.moduli.pop_back();
		assert (text.moduli.size() > 0); // the last prime can't be popped
	}
	void mod_switch_to(Text& text, EncryptionParameters parms, Text& dest) {
		assert(text.nb_slots*2 == parms.N);
		assert(text.moduli.size() >= parms.moduli.size());

		dest = Text(text);

		while (dest.moduli.size() > parms.moduli.size()) {
			dest.moduli.pop_back();
		}
		assert (dest.moduli.size() > 0); // the last prime can't be popped
	}
	void mod_switch_to_inplace(Text& text, EncryptionParameters parms) {
		mod_switch_to(text, parms, text);
	}

	// Relin
	void relinearize_inplace(Text& text, RelinKeys& k) {
		// noop
		loc[4]++; glob[4]++;
	}

	//vector<int> moduli;
	//int nb_slots;
};
	
} // namespace seal
