#!/usr/bin/env bash

all_data=(false true)
length=(60 120 180 240 300)  # (120 180 240)
store_id=("store_A" "store_B" "store_C" "store_D" "store_E")

for opt1 in "${all_data[@]}"; do
    for opt3 in "${length[@]}"; do
        for opt4 in "${store_id[@]}"; do
            echo "!!!!here!!!!!"
            echo "train_epochs = 1"
            echo "all_data = $opt1"
            echo "length = $opt3"
            echo "store_id = $opt4"
            echo "!!!!!!!!!!!!!!!!!"
            python data.py --all_data=${opt1} --train_epochs=1 --training_length=$opt3 --store_id=$opt4
            python data.py --all_data=${opt1} --train_epochs=1 --training_length=$opt3 --store_id=$opt4
            python data.py --all_data=${opt1} --train_epochs=1 --training_length=$opt3 --store_id=$opt4
            python data.py --all_data=${opt1} --train_epochs=1 --training_length=$opt3 --store_id=$opt4
            python data.py --all_data=${opt1} --train_epochs=1 --training_length=$opt3 --store_id=$opt4
        done
    done
done


