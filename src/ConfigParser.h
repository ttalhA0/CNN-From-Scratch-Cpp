#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <fstream>
#include <stdexcept>
#include <sstream>
#include <string>

struct ConfigParser
{
    double learning_rate;
    int epochs;
    int num_filters;
    int filter_size;
    int pool_size;
    int stride;
    int hidden_nodes;
    int output_nodes;
    
    // It is computed automatically
    int flatten_size;

    void load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open())
            throw std::runtime_error("The Config File cannot be found: " + filename);

        std::string line;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream iss(line);
            std::string key;
            iss >> key;

            if (key == "learning_rate") iss >> learning_rate;
            else if (key == "epochs") iss >> epochs;
            else if (key == "num_filters") iss >> num_filters;
            else if (key == "filter_size") iss >> filter_size;
            else if (key == "pool_size") iss >> pool_size;
            else if (key == "stride") iss >> stride;
            else if (key == "hidden_nodes") iss >> hidden_nodes;
            else if (key == "output_nodes") iss >> output_nodes;
        }

        int conv_out_size = 28 - filter_size + 1;
        int pool_out_size = (conv_out_size - pool_size) / stride + 1;
        flatten_size = num_filters * pool_out_size * pool_out_size;
    }
};


#endif