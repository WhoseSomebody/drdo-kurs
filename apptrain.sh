#!/bin/bash

while [[ $# > 1 ]]
do
key="$1"

case $key in
    -i|--input)
    INPUT="$2"
    shift # past argument
    ;;
    -o|--output)
    OUTPUT="$2"
    shift # past argument
    ;;
    *)
            # unknown option
    ;;
esac
shift # past argument or value
done
if [ -z ${OUTPUT+x} ]; then OUTPUT=util/output.txt  ; else OUTPUT=util/$OUTPUT; fi
echo INPUT FILE  = "${INPUT}" WITH EXTENSION "${INPUT##*.}"
echo "RUNNING FIRST PYTHON SCRIPT (convertion binary into text file ${OUTPUT})"

# convert to *.wav if needed and to *.txt
python util/from_bin.py $INPUT $OUTPUT
CUR_OUT=$(ls ${OUTPUT}_*)
echo $CUR_OUT

# get the number of frames in audio
FRAMES=${CUR_OUT##*_}
echo $FRAMES

echo PREPARING WORKING DIRECTORIES...
NEWDIR=$(date +%Y.%m.%d_\(%H:%M\))
mkdir data/$NEWDIR
mv $CUR_OUT data/$NEWDIR/input.txt

echo TRAIN STARTED
# To train NN again.
th train.lua -data_dir data/$NEWDIR -opencl 1


echo "THANKS FOR USING OUR APP :)"

