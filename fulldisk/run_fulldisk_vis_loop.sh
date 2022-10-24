#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`
cd /home/poker/goes16/fulldisk

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /weather/data/goes16/TIRS/02/latest.nc /dev/shm/latest_TIRS_02.nc
cmp /weather/data/goes16/TIRS/02/latest.nc /dev/shm/latest_TIRS_02.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /weather/data/goes16/TIRS/02/latest.nc /dev/shm/latest_TIRS_02.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /weather/data/goes16/TIRS/02/latest.nc /dev/shm/latest_TIRS_02.nc
  sleep 028
  echo 'start fulldisk visible at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_fulldisk_visible_v2.py /dev/shm/latest_TIRS_02.nc
  echo 'start fulldisk visible sqrt at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_fulldisk_visible_sqrt_v2.py /dev/shm/latest_TIRS_02.nc
  cmp /weather/data/goes16/TIRS/02/latest.nc /dev/shm/latest_TIRS_02.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


python goes16_conusc_wvl.py $time


