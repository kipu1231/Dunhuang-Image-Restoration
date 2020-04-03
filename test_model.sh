#!/usr/bin/env bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pLpA3TioffRXUSBRk5NybcUOLWvT2r8R' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pLpA3TioffRXUSBRk5NybcUOLWvT2r8R" -O PartConvModel/Net_best_fine_tune.pth-6.tar && rm -rf /tmp/cookies.txt
RESUME='PartConvModel/Net_best_fine_tune.pth-6.tar'
python3 PartConvModel/test.py --resume $RESUME --data_dir_test $1 --predictions $2