for i in $(seq 0 100);
do
    "/mnt/c/Users/benoit.alcaraz/AppData/Local/Programs/Python/Python39/python.exe" src/main.py --considerations 50 --noise $i --function 2 >"noise_${i}_function_2" &
done
