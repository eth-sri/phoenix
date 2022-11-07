# Adult-MockSEAL
for i in {0..15059}
do 
    cmd="./mockphoenix configs/adult_mlp2.json $i"
    $cmd
done