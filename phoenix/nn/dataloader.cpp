#include "dataloader.h"

void Dataloader::get_input(int idx, std::vector<double>& dest, bool div) {
    if (random) {
        dest = get_random_vector(input_size);
        return;
    }
    std::stringstream input_key;
    input_key << "input" << idx;
    dest = test_csv->GetRow<double>(input_key.str());
    if (div) {
        for (int i = 0; i < dest.size(); i++) {
            dest[i] /= 255.0;
        }
    }
    dest.resize(input_size, 0); // must pad with 0s
}


int Dataloader::get_target(int idx) {
    if (random) {
        return rand() % 10;
    }
    std::stringstream target_key;
    target_key << "target" << idx;
    int target = test_csv->GetRow<int>(target_key.str())[0];
    return target;
}