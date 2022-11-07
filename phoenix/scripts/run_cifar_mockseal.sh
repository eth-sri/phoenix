# CIFAR-MockSEAL
for i in {0..9999}
do 
    cmd="./mockphoenix configs/cifar_mlp3.json $i"
    $cmd
done