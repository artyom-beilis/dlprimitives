#include <dlprim/net.hpp>
#include <fstream>
#include <iostream>

namespace dp = dlprim;

class MnistReader {
public:
    MnistReader(std::string const &img,std::string const &lbl)
    {
        img_.open(img,std::fstream::binary);
        lbl_.open(lbl,std::fstream::binary);
        if(!img_ || !lbl_) {
            throw std::runtime_error("Failed to open mnsit db");
        }
        char tmp[16];
        lbl_.read(tmp,8);
        img_.read(tmp,16);
    }
    int get_batch(int *lables,float *data,int max)
    {
        int read = 0;
        constexpr int img_size = 28*28;
        constexpr float factor = 1.0f/255.0f;
        unsigned char buf[img_size];
        char lbl;
        while(max > 0 && !img_.eof()) {
            img_.read((char*)buf,img_size);
            lbl_.read(&lbl,1);
            if(img_.gcount()!=img_size || lbl_.gcount() != 1)
                break;
            for(int i=0;i<img_size;i++) {
                *data++ = buf[i] * factor;
            }
            *lables++ = lbl;
            max--;
            read++;
        }
        return read;
    }

private:
    std::ifstream img_,lbl_;
};

int argmax(float *p,int n)
{
    int r = 0;
    for(int i=1;i<n;i++){
        if(p[i] > p[r])
            r = i;
    }
    return r;
}

int main(int argc,char **argv)
{
    if(argc!=6) {
        std::cerr << "Usage device net.json net.h5 minst_data mnist_labels" << std::endl;
        return 1;
    }
    dp::Context ctx(argv[1]);
    std::cout << "Using: " << ctx.name() << std::endl;
    
    std::string net_js = argv[2];
    std::string net_h5 = argv[3];
    std::string mnist_img = argv[4];
    std::string mnist_lbl = argv[5];

    MnistReader reader(mnist_img,mnist_lbl);
    dp::Net net(ctx);
    net.load_from_json_file(net_js);
    net.setup();
    net.load_parameters_from_hdf5(net_h5);
    net.copy_parameters_to_device();
    dp::Tensor data = net.tensor("data");
    dp::Tensor prob = net.tensor("prob");
    int batch = data.shape()[0];
    std::vector<int> labels(batch);
    int n;
    cl::CommandQueue q=ctx.make_queue();
    int correct = 0;
    int total = 0;
    double total_time = 0;
    while((n = reader.get_batch(labels.data(),data.data<float>(),batch)) > 0) {
        if(n != batch) {
            data.reshape(dp::Shape(n,1,28,28));
            net.reshape();
        }
        auto start = std::chrono::system_clock::now();
        data.to_device(q,false);
        net.forward(q);
        prob.to_host(q,true);
        auto stop = std::chrono::system_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::duration<double> > ((stop-start)).count();
        for(int i=0;i<n;i++) {
            int pred = argmax(prob.data<float>() + 10*i,10);
            #ifdef PRINT_LOG
                for(int r=0;r<28;r++) {
                    for(int c=0;c<28;c++) {
                        std::cout << (data.data<float>()[28*28*i + 28*r + c] > 0.2 ? 'X' : ' ');
                        //std::cout << (data.data<float>()[28*28*i + 28*r + c]) << " ";
                    }
                    std::cout << std::endl;
                }
                for(int j=0;j<10;j++)
                    std::cout << prob.data<float>()[10*i + j] << " ";
                std::cout << "->" << pred << " exp " << labels[i] << std::endl;
            #endif            
            if(pred == labels[i])
                correct++;
        }
        total += n;
        if(n < batch)
            break;
    }
    std::cout << "Correct " << correct << " out of " << total << " = " << (100.0 * correct / total) <<"%" << std::endl;
    std::cout << "Time per sample: " << (total_time / total * 1e6) << "us" << std::endl;

}
