#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

cd /home/poker/goes16/fulldiskc_grb

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /home/ldm/data/grb/fulldisk/09/latest.nc /dev/shm/latest_GRB_FULLDISK_09.nc
cmp /home/ldm/data/grb/fulldisk/09/latest.nc /dev/shm/latest_GRB_FULLDISK_09.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/fulldisk/09/latest.nc /dev/shm/latest_GRB_FULLDISK_09.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /home/ldm/data/grb/fulldisk/09/latest.nc /dev/shm/latest_GRB_FULLDISK_09.nc
  sleep 100
  echo 'start fulldisk wvc at ' `date`
# Color
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_fulldisk_wvc.py /dev/shm/latest_GRB_FULLDISK_09.nc
  echo 'end   fulldisk wvc at ' `date`
  echo 'start fulldisk wv b/w at ' `date`
# B/W
  /home/poker/miniconda3/envs/goes16_201710/bin/python /home/poker/goes16/fulldisk_grb/goes16_GRB_fulldisk_wv.py /dev/shm/latest_GRB_FULLDISK_09.nc
  echo 'end   fulldisk wv  at ' `date`

# goes16_GRB_fulldisk_IR09_irc.py goes16_GRB_fulldisk_IR09_ircm.py
  cmp /home/ldm/data/grb/fulldisk/09/latest.nc /dev/shm/latest_GRB_FULLDISK_09.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


python goes16_conusc_ircm.py $time


