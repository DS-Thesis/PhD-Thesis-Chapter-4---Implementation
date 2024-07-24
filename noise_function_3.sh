for i in $(seq 0 100);
do
    python src/main.py --considerations 50 --noise $i --function 3 >"noise_${i}_function_3" &
done
