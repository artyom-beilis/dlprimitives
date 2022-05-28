///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#pragma once
#include <dlprim/json.hpp>
#include <fstream>
#include <sstream>
#include <iostream>


namespace dp = dlprim;
///
/// Custom configuration for predict, mean/std normalization if needed
/// class names for output
///
struct Config {
    std::vector<float> fact;
    std::vector<float> off;
    std::vector<std::string> class_names;
    int c=-1,h=-1,w=-1;
    int classes=-1;
    void load(std::string const &path)
    {
        if(path.empty()) {
            fact.resize(c,1/255.0f);
            off.resize(c,0.0f);
            return;
        }
        std::ifstream f(path);
        dp::json::value v;
        if(!v.load(f,true)) {
            throw std::runtime_error("Failed to read: " + path);
        }
        if(!v.find("mean").is_undefined()) {
            auto mean = v.get<std::vector<float> >("mean");
            auto std = v.get<std::vector<float> >("std");
            fact.resize(mean.size());
            off.resize(mean.size());
            for(unsigned i=0;i<mean.size();i++) {
                fact[i] = 1.0 / 255.0f / std[i];
                off[i] = -mean[i] / std[i];
            }
            if(int(fact.size()) != c) {
                std::cerr << "Norm " << fact.size() << " exp " << c << std::endl;
                throw std::runtime_error("Invalid normalization size");
            }
        }
        if(!v.find("class_names").is_undefined()) {
            class_names = v.get<std::vector<std::string> >("class_names");
            if(int(class_names.size()) != classes) {
                throw std::runtime_error("Names of classes does not match network");
            }
        }
    }
};

///
/// Copy PPM/PGM image to tensor and convert
///
inline void load_image(dp::Tensor &t,int batch,std::string const &path,Config const &cfg)
{
    float *img = t.data<float>() + t.shape().size_no_batch() * batch;
    std::ifstream f(path,std::ifstream::binary);
    std::string header;
    f >> header;
    if(header == "P5") {
        if(cfg.c != 1)
            throw std::runtime_error("PGM requires 1 channel: " + path);
    }
    else if(header == "P6") {
        if(cfg.c != 3) {
            throw std::runtime_error("PPM requires 3 channel: " + path);
        }
    }
    else {
        throw std::runtime_error("Unspoorted format " + path);
    }
    std::string data,line;
    int w=-1,h=-1,bits=-1;
    while(std::getline(f,line)) {
        if(line.c_str()[0] == '#')
            continue;
        data += ' ';
        data += line;
        std::istringstream ss(data);
        w=h=bits = -1;
        ss >> w >> h >> bits;
        if(bits != -1)
            break;
    }
    if(bits != 255 || w <=0 || h <= 0) {
        throw std::runtime_error("Failed to parse " + path);
    }
    std::vector<unsigned char> buf(w*h*cfg.c);
    f.read(reinterpret_cast<char*>(buf.data()),buf.size());
    if(!f) {
        throw std::runtime_error("Failed to read " + path);
    }
    if(w < cfg.w || h < cfg.h) {
        throw std::runtime_error("image too small " + path);
    }
    int dc = (w - cfg.w)/2;
    int dr = (h - cfg.h)/2;
    for(int chan = 0;chan < cfg.c ;chan ++) {
        for(int r=0;r<cfg.h;r++) {
            for(int c=0;c<cfg.w;c++) {
                float v = cfg.fact[chan] * buf[(r+dr) * w * cfg.c + (c+dc) * cfg.c + chan] + cfg.off[chan];
                *img++ = v;
            }
        }
    }
}
inline int argmax(float *p,int n)
{
    int r = 0;
    for(int i=1;i<n;i++){
        if(p[i] > p[r])
            r = i;
    }
    return r;
}

