#!/bin/bash

export PATH="/home/poker/miniconda3/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

export time=`ls -1 /weather/data/goes16/TIRE/14/*PAA.nc | awk '{$1 = substr($1,30,12)} 1' | sort -u | tail -2 | head -1`

sleep 74

echo $time

cd /home/poker/goes16/conusc

python goes16_conusc_ircm.py $time


