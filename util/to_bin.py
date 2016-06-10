#!/usr/bin/python
import sys, os.path
import binascii
from pydub import AudioSegment


input_f = str(sys.argv[1])
output_f = str(sys.argv[2]) if len(sys.argv) > 2 else "../output.wav"

print [input_f, output_f]
if not os.path.isfile(str(sys.argv[1])):
    sys.exit("Input file '" + str(sys.argv[1]) + "' doesn't exist!") 

with open(input_f) as f, open(output_f, 'wb') as fout:
    for line in f:
        fout.write(
            binascii.unhexlify(''.join(line.split()))
        )
    print ("Successfully created audio file '" + output_f +"'!")

print ("Converting to mp3...")
AudioSegment.from_wav(output_f).export(output_f.split('.wav')[0]+".mp3", format="mp3")
print ("Successfully converted to " + output_f.split('.wav')[0] + ".mp3")



# with open(input_f) as f, open(output_f, 'wb') as fout:
#     for line in f:
#     	print len(line)
#         fout.write(
#             binascii.unhexlify(line)
#         ) 

#     print ("Successfully created audio file '" + output_f +"'!")
