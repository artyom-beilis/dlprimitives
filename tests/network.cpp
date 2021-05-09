namespace dl = dlprim;
int main(int argc,char **argv)
{
    int dev_id = -1;
    if(argc >= 2) {
        dev_id = atoi(argv[1]);
    }
    dl::Context ctx(dev_id);
    dl::Net net(ctx);
    net.add_input_tensor("data",dl::Shape(8,1,28,28));
    net.add_sequential_operator(dl::Flatten::create(ctx),"flatten");
    net.add_sequential_operator(dl::InnerProduct::create(ctx,dl::InnerProduct(InnerProductConfig(28*28,512,dl::StandardActivations.relu))),"ip1");
    net.add_sequential_operator(dl::InnerProduct::create(ctx,dl::InnerProduct(InnerProductConfig(512,10))),"ip2");
    net.add_sequential_operator(dl::InnerProduct::create(ctx,dl::SoftMax::create(ctx),"prob");
    net.build(dl::
}
