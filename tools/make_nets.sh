device=hd530
pl=1
dldev=$pl:0
batch=8
PYTHON=python3
mkdir -p nets log
for net in alexnet resnet18 resnet50 vgg16 mobilenet_v2
do
    suffix=$net-$device-$batch
    base_name=nets/${net}-b$batch
    echo $net
    if false
    then
        if false
        then
            #$PYTHON ../tools/validate_network.py --model $net --export $base_name.onnx --train --batch $batch
            echo "Train PT"
            $PYTHON ../tools/validate_network.py --model $net --benchmark --train --batch $batch | tee log/pt-$net-$device-train-$batch.txt | tail -n 5
            echo "Test PT"
            $PYTHON ../tools/validate_network.py --model $net --benchmark --batch $batch | tee log/pt-$net-$device-test-$batch.txt | tail -n 5
            $PYTHON ../tools/onnx2dp.py --model $base_name.onnx --js $base_name.js --h5 $base_name.h5 >/dev/null
        fi
        echo "Valid"
        ./image_predict -o../tools/imagenet_predict_config.json $dldev $base_name.js $base_name.h5 ../tests/samples/*.ppm | tee log/$net-$device-dl-predict-$batch.txt
        echo "Train DL"
        ./benchmark -b $dldev $base_name.js $base_name.h5 | tee log/dl-$net-train-$device-$batch.txt | tail -n 5
        echo "Test DL"
        ./benchmark $dldev $base_name.js $base_name.h5 | tee log/dl-$net-test-$device-$batch.txt | tail -n 5
    fi
    $PYTHON ../tools/keras_benchmark.py --warm 3 --iters 5 --model $base_name.js --batch $batch | tee log/plaidml-$net-$device-$batch.txt
    #/home/artik/Packages/caffe/caffe_ocl_old/build_clblas/tools/caffe time -model $base_name.prototxt -gpu $pl -iterations 25 -phase TRAIN 2>&1 |  tee log/caffe-$net-$device-$batch-train.txt
    #/home/artik/Packages/caffe/caffe_ocl_old/build_clblas/tools/caffe time -model $base_name.prototxt -gpu $pl -iterations 25 -phase TEST 2>&1 |  tee log/caffe-$net-$device-$batch-test.txt
done
     
