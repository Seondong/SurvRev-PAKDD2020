#!/usr/bin/env bash

all_data=(1000 5000 50000)
num_hist=(5)  # (4 5 6 7 8)
length=(180 240 120 60)  # (120 180 240)
store_id=("store_A" "store_B" "store_C" "store_D" "store_E")
previous_visits=(true false)

for opt1 in "${all_data[@]}"; do
    if opt1 == 50000; then   # all_data=true part
        for opt2 in "${num_hist[@]}"; do
            for opt3 in "${length[@]}"; do
                for opt4 in "${store_id[@]}"; do
                    for opt5 in "${previous_visits[@]}"; do
                        echo "!!!!here!!!!!"
                        echo "train_epochs = 1"
                        echo "all_data = $opt1"
                        echo "max_num_histories = $opt2"
                        echo "length = $opt3"
                        echo "store_id = $opt4"
                        echo "previous_visits = $opt5"
                        echo "!!!!!!!!!!!!!!!!!"
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5 --train_epochs=1
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5 --train_epochs=1
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5 --train_epochs=1
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5 --train_epochs=1
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5 --train_epochs=1
                    done
                done
            done
        done
    else
        for opt2 in "${num_hist[@]}"; do
            for opt3 in "${length[@]}"; do
                for opt4 in "${store_id[@]}"; do
                    for opt5 in "${previous_visits[@]}"; do
                        echo "!!!!!there!!!!!"
                        echo "all_data = $opt1"
                        echo "train_epochs = 10"
                        echo "max_num_histories = $opt2"
                        echo "length = $opt3"
                        echo "store_id = $opt4"
                        echo "previous_visits = $opt5"
                        echo "!!!!!!!!!!!!!!!!!"
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5
                        python survrevtensorflow2.py --all_data=${opt1} --max_num_histories=$opt2 --training_length=$opt3 --store_id=$opt4 --previous_visits=$opt5
                    done
                done
            done
        done
    fi
done


