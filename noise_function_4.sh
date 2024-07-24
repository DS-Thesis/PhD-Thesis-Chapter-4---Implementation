for i in $(seq 0 100);
do
    python src/main.py --considerations 50 --noise $i --function 4 >"noise_${i}_function_4" &
done
