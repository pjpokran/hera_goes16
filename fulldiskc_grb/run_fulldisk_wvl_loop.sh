#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

cd /home/poker/goes16/fulldiskc_grb

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /home/ldm/data/grb/fulldisk/10/latest.nc /dev/shm/latest_GRB_FULLDISK_10.nc
cmp /home/ldm/data/grb/fulldisk/10/latest.nc /dev/shm/latest_GRB_FULLDISK_10.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/fulldisk/10/latest.nc /dev/shm/latest_GRB_FULLDISK_10.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /home/ldm/data/grb/fulldisk/10/latest.nc /dev/shm/latest_GRB_FULLDISK_10.nc
  sleep 210
  echo 'start fulldisk wvl at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_fulldisk_wvl.py /dev/shm/latest_GRB_FULLDISK_10.nc
  echo 'end   fulldisk wvl at ' `date`

# goes16_GRB_fulldisk_IR10_irc.py goes16_GRB_fulldisk_IR10_ircm.py
  cmp /home/ldm/data/grb/fulldisk/10/latest.nc /dev/shm/latest_GRB_FULLDISK_10.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


python goes16_conusc_ircm.py $time


