#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin:$PATH"

cd /home/poker/goes16/conusc

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRE/09/latest.nc /dev/shm/latest_TIRE_09.nc
cmp /weather/data/goes16/TIRE/09/latest.nc /dev/shm/latest_TIRE_09.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRE/09/latest.nc /dev/shm/latest_TIRE_09.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRE/09/latest.nc /dev/shm/latest_TIRE_09.nc
  sleep 49
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_conusc_wv_fixeddisk.py /dev/shm/latest_TIRE_09.nc
  cmp /weather/data/goes16/TIRE/09/latest.nc /dev/shm/latest_TIRE_09.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


#python goes16_conusc_wv.py $time


