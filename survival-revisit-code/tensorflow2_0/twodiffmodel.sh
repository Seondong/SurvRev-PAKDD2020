#!/usr/bin/env bash

all_data=(false true)
num_hist=(5)  # (4 5 6 7 8)
length=(240)  # (120 180 240)
store_id=("store_A" "store_B" "store_C" "store_D" "store_E")
previous_visits=(false true)

for opt5 in "${previous_visits[@]}"; do
    if ${opt5}; then   # all_data=true part
        python survrevtensorflow2.py --previous_visits=${opt5}
    else
        python survrevtensorflow2.py --previous_visits=${opt5}
    fi
done