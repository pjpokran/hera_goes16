#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin/:$PATH"

cd /home/poker/goes16/conusc

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRE/07/latest.nc /dev/shm/latest_TIRE_07.nc
cmp /weather/data/goes16/TIRE/07/latest.nc /dev/shm/latest_TIRE_07.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRE/07/latest.nc /dev/shm/latest_TIRE_07.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRE/07/latest.nc /dev/shm/latest_TIRE_07.nc
  sleep 65
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_conusc_swir_fixeddisk.py /dev/shm/latest_TIRE_07.nc
  cmp /weather/data/goes16/TIRE/07/latest.nc /dev/shm/latest_TIRE_07.nc > /dev/null
  CONDITION=$?
#  echo repeat

done




# /home/poker/goes16/conusc/goes16_conusc_swir.py

