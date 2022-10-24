#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "13 min ago"`
export time=`ls -1 /weather/data/goes16/TIRS/09/*PAA.nc | awk '{$1 = substr($1,30,12)} 1' | sort -u | tail -2 | head -1`


echo $time

sleep 120

cd /home/poker/goes16/fulldisk

python goes16_fulldisk_wv.py $time


