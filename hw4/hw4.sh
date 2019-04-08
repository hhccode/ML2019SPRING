#!/bin/bash
wget 'https://www.dropbox.com/s/vd5j7qtbv5y11a6/bestmodel.pkl?dl=1'
python3 hw4_1.py $1 $2
python3 hw4_2.py $1 $2
python3 hw4_3.py $1 $2