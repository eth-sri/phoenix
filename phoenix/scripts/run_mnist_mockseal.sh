# MNIST-MockSEAL
for i in {0..9999}
do 
    cmd="./mockphoenix configs/mnist_mlp2.json $i"
    $cmd
done