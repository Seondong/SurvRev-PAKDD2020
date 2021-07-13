#!/usr/bin/env bash

all_data=(false)
num_hist=(5)  # (4 5 6 7 8)
length=(180 240)  # (120 180 240)
store_id=("store_A" "store_B" "store_C" "store_D" "store_E")

for opt1 in "${all_data[@]}"; do
    if ${opt1}; then   # all_data=true part
        for opt2 in "${num_hist[@]}"; do
            for opt3 in "${length[@]}"; do
                for opt4 in "${store_id[@]}"; do
                    echo "!!!!here!!!!!"
                    echo "train_epochs = 1"
                    echo "all_data = $opt1"
                    echo "max_num_histories = $opt2"
                    echo "length = $opt3"
                    echo "store_id = $opt4"
                    echo "!!!!!!!!!!!!!!!!!"
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                done
            done
        done
    else
        for opt2 in "${num_hist[@]}"; do
            for opt3 in "${length[@]}"; do
                for opt4 in "${store_id[@]}"; do
                    echo "!!!!!there!!!!!"
                    echo "all_data = $opt1"
                    echo "train_epochs = 10"
                    echo "max_num_histories = $opt2"
                    echo "length = $opt3"
                    echo "store_id = $opt4"
                    echo "!!!!!!!!!!!!!!!!!"
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                    python survrevtensorflow2.py --gpu_id=5 --all_data=${opt1} --train_epochs=10 --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4
                done
            done
        done
    fi
done


