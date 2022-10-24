#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

cd /home/poker/goes16/conusc

cp /weather/data/goes16/TIRE/05/latest.nc /dev/shm/latest_TIRE_05.nc
cmp /weather/data/goes16/TIRE/05/latest.nc /dev/shm/latest_TIRE_05.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRE/05/latest.nc /dev/shm/latest_TIRE_05.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRE/05/latest.nc /dev/shm/latest_TIRE_05.nc
  sleep 222
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_conusc_snowice_fixeddisk.py /dev/shm/latest_TIRE_05.nc
  cmp /weather/data/goes16/TIRE/05/latest.nc /dev/shm/latest_TIRE_05.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


#python goes16_conusc_snowice.py $time


