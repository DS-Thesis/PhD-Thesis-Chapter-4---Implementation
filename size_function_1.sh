for i in $(seq 0 100);
do
    python src/main.py --considerations $i --noise 0 --function 1 >"size_${i}_function_1" &
done
