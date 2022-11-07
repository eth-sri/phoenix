#pragma once

#include <cstddef>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <string>
#include <random>
#include <memory>
#include <algorithm>
#include <chrono>
#include "utils/jsoncpp/json-forwards.h"
#include "utils/jsoncpp/json.h"

#ifdef MOCK
    #include "utils/mock_seal.h"
#else
    #include "seal/seal.h"
#endif

typedef std::chrono::high_resolution_clock Clock;

inline std::vector<int> parse_int_vector(Json::Value& arr) {
	std::vector<int> ret;
	for (auto& x : arr) {
		ret.push_back(x.asInt());
	}
	return ret;
}

inline std::vector<double> parse_double_vector(Json::Value& arr) {
	std::vector<double> ret;
	for (auto& x : arr) {
		ret.push_back(x.asDouble());
	}
	return ret;
}

inline Json::Value load_config(char* config_path) {
	Json::Value root;
  	std::ifstream ifs;
  	ifs.open(config_path);

  	Json::CharReaderBuilder builder;
  	builder["collectComments"] = false;
  	JSONCPP_STRING errs;
  	if (!parseFromStream(builder, ifs, &root, &errs)) {
    	std::cout << errs << std::endl;
		assert(false);
  	}
	return root;
}

inline bool ispow2(int x) {
	while (x % 2 == 0) x /= 2;
	return (x == 1);
}

inline void print_minmax(std::vector<double>& z) {
	double mini = 1;
	double maxi = -1;
	for (double x : z) {
		mini = std::min(mini, x);
		maxi = std::max(maxi, x);
	}
	std::cout << "Min = " << mini << " | Max = " << maxi << std::endl;
}

// nb_slots divided into batch_size blocks
// each block has hb_batches small blocks of size 32, first 10 are predictions
inline void extend_mask_to_small_blocks(std::vector<double>& mask, int nb_slots, int block_sz, int miniblock_sz, int nb_miniblocks) {
  int sz = mask.size();
  mask.resize(nb_slots, 0);

  int nb_blocks = nb_slots / block_sz;
  // max nb_miniblocks is block_sz / miniblock_sz but we care about nb_miniblocks

  for (int block = 0; block < nb_blocks; block++) {
	for (int miniblock = 0; miniblock < nb_miniblocks; miniblock++) {
	  for (int j = 0; j < sz; j++) {
	    mask[block*block_sz + miniblock*miniblock_sz + j] = mask[j];
	  }
	}
  }
}

// nb_slots divided into batch_size blocks
inline void extend_mask_to_blocks(std::vector<double>& mask, int nb_slots, int block_sz) {
  int sz = mask.size();
  mask.resize(nb_slots, 0);

  int nb_blocks = nb_slots / block_sz;

  for (int block = 1; block < nb_blocks; block++) {
    for (int j = 0; j < sz; j++) {
	  mask[block*block_sz+j] = mask[j];
	}
  }
}

inline void repeat(const std::vector<double>& src, std::vector<double>& dst, int k) {
  int sz = src.size();
  dst.reserve(k*sz);
  while (k--) {
    dst.insert(dst.end(), src.begin(), src.end());
  }
}

inline int argmax(const std::vector<double>& logits, int nb_classes) {
	int best_idx = 0;
	for (int i = 1; i < nb_classes; i++) {
		if (logits[i] > logits[best_idx]) {
			best_idx = i;
		}
	}
	return best_idx;
}

inline void print_logits_batched(int to_print, const std::vector<std::vector<double>>& all_logits, int nb_classes, int target) {
  std::cout << "printing first " << to_print << " logit arrays" << std::endl;
  for (int i = 0; i < to_print; i++) {
	const std::vector<double>& logits = all_logits[i];

	std::stringstream ss;

	std::cout << "logits: [";
	int pred = 0;
	double max_score = (double)logits[0];
	for (int i = 0; i < nb_classes; i++) {
		if (logits[i] > max_score) {
		max_score = logits[i];
		pred = i;
		}
		
		std::cout << (double) logits[i];

		if (i < nb_classes-1) std::cout << " ";
		else std::cout << "]";
		if (i == 4) std::cout << "| ";
	}
	std::cout << ", pred: " << pred;
	if (pred == target) std::cout << " (OK)";
	else std::cout << " (WRONG)";
	std::cout << std::endl;
  }
}

inline void print_logits_batched(const std::vector<double>& logits, int nb_logits, int nb_slots, int block_sz, int target, int to_print) {
  std::cout << "printing first " << to_print << " logit arrays" << std::endl;
  for (int i = 0; i < to_print; i++) {
	int off = block_sz*i;

	std::stringstream ss;

	std::cout << "logits: [";
	int pred = 0;
	double max_score = (double)logits[0];
	for (int i = 0; i < nb_logits; i++) {
		if (logits[off+i] > max_score) {
		max_score = logits[off+i];
		pred = i;
		}
		
		std::cout << (double) logits[off+i];

		if (i < nb_logits-1) std::cout << " ";
		else std::cout << "]";
		if (i == 4) std::cout << "| ";
	}
	std::cout << ", pred: " << pred;
	if (pred == target) std::cout << " (OK)";
	else std::cout << " (WRONG)";
	std::cout << std::endl;
  }
}

inline void print_logits(const std::vector<double>& logits, int nb_logits, int target) {
  std::stringstream ss;

  std::cout << "logits: [";
  int pred = 0;
  double max_score = (double)logits[0];
  for (int i = 0; i < nb_logits; i++) {
    if (logits[i] > max_score) {
      max_score = logits[i];
      pred = i;
    }
    
    std::cout << (double) logits[i];

    if (i < nb_logits-1) std::cout << " ";
    else std::cout << "]";
    if (i == 4) std::cout << "| ";
  }
  std::cout << ", pred: " << pred;
  if (pred == target) std::cout << " (OK)";
  else std::cout << " (WRONG)";
  std::cout << std::endl;
}

inline void log_time(std::stringstream &ss_time, std::string key, std::chrono::time_point<Clock> t1, std::chrono::time_point<Clock> t2) {
  double duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << key << ": " << duration << "ms" << std::endl;
  ss_time << key << ": " << duration << "ms" << "\n";
}