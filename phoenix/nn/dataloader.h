#pragma once

#include <vector>
#include <cassert>
#include <memory>
#include <string>
#include "utils/rapidcsv.h"
#include "linalg/linalg.h"

class Dataloader {
public:
    Dataloader(const std::string& dataset, const std::string& data_path, int input_size, bool rand_data) {
        if (rand_data) {
            std::cout << "Random data!" << std::endl;
            this->random = true;
            this->input_size = input_size;
            return;
        }
        this->random = false;
        
        test_csv = std::make_unique<rapidcsv::Document>(data_path, rapidcsv::LabelParams(-1, 0));

        this->input_size = input_size;
    }

    Dataloader(int input_size) {
        this->random = true;
        this->input_size = input_size;
    }

    void get_input(int idx, std::vector<double>& dest, bool div);
    int get_target(int idx);
    
private:
    bool random; // use random data

    int input_size;
    std::unique_ptr<rapidcsv::Document> test_csv;
};
