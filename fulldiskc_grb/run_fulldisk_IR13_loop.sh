#!/bin/bash

export PATH="/home/poker/miniconda3/envs/goes16_201710/bin:$PATH"
#export time=`date -u "+%Y%m%d%H%M" -d "6 min ago"`

cd /home/poker/goes16/fulldiskc_grb

#  /weather/data/goes16/"+prod_id+"/"+band+"/latest.nc
cp /home/ldm/data/grb/fulldisk/13/latest.nc /dev/shm/latest_GRB_FULLDISK_13.nc
cmp /home/ldm/data/grb/fulldisk/13/latest.nc /dev/shm/latest_GRB_FULLDISK_13.nc > /dev/null
CONDITION=$?
#echo $CONDITION

while :; do

  until [ $CONDITION -eq 1 ] ; do
#     echo same
     sleep 5
     cmp /home/ldm/data/grb/fulldisk/13/latest.nc /dev/shm/latest_GRB_FULLDISK_13.nc > /dev/null
     CONDITION=$?
  done

#  echo different
  cp /home/ldm/data/grb/fulldisk/13/latest.nc /dev/shm/latest_GRB_FULLDISK_13.nc
  sleep 10
# Color AWIPS
  echo 'start fulldisk IR13 irc at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_fulldisk_IR13_irc.py /dev/shm/latest_GRB_FULLDISK_13.nc
  echo 'end   fulldisk IR13 irc  at ' `date`
# Color McIDAS
  echo 'start fulldisk IR13 ircm at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python goes16_GRB_fulldisk_IR13_ircm.py /dev/shm/latest_GRB_FULLDISK_13.nc
  echo 'end   fulldisk IR13 ircm at ' `date`
# B/W
  echo 'start fulldisk IR13 b/w at ' `date`
  /home/poker/miniconda3/envs/goes16_201710/bin/python /home/poker/goes16/fulldisk_grb/goes16_GRB_fulldisk_IR13_ir.py /dev/shm/latest_GRB_FULLDISK_13.nc
  echo 'end   fulldisk IR13 b/w  at ' `date`

# goes16_GRB_fulldisk_IR13_irc.py goes16_GRB_fulldisk_IR13_ircm.py
  cmp /home/ldm/data/grb/fulldisk/13/latest.nc /dev/shm/latest_GRB_FULLDISK_13.nc > /dev/null
  CONDITION=$?
#  echo repeat

done


python goes16_conusc_ircm.py $time


