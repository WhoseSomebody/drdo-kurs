
# import sys, os.path
# from pydub import AudioSegment
# import binascii
# import wave
# import contextlib

# def read_file(inf = '', ouf = ''):
# 	input_f = inf
# 	output_f = ouf
# 	if not os.path.isfile(inf):
# 	    sys.exit("Input file '" + inf + "' doesn't exist!") 

# def format_detect(input_f = ''):
# 	if input_f.split('.')[-1] == 'mp3':
# 		print 'mp3 convertion to wav...'
# 		sound = AudioSegment.from_mp3(input_f)
# 		sound.export("temp.wav", format="wav"); input_f = "temp.wav"
# 	elif input_f.split('.')[-1] == 'wav':
# 		print 'already *.wav format'
# 	else:
# 		sys.exit("NOT AUDIO INPUT. (choose *.mp3 or *.wav)")

# def convertion_to_text(input_f = '', output_f = ''):
# 	chunk_size = 1024
# 	with open(input_f, 'rb') as f, open(output_f+"_"+str(os.path.getsize(input_f)), 'w') as fout:
# 	    while True:
# 	        data = f.read(chunk_size)
# 	        if not data:
# 	            break
# 	        hexa = binascii.hexlify(data)
# 	        hexa_string = hexa.decode('ascii')
# 	        fout.write(hexa_string) 
# 	    print("converted to text in '" + output_f + "'")





#!/usr/bin/python

import sys, os.path
from pydub import AudioSegment
import binascii
import wave
import contextlib

inp = ''
out = ''
try:
	inp = str(sys.argv[1])
	out = str(sys.argv[2]) if len(sys.argv) > 2 else "output.txt"
except:
	print "Error"

def read_file(inpf = inp, ouf = out):
	input_f = inpf
	output_f = ouf
	if not os.path.isfile(inpf):
	    sys.exit("Input file '" + inpf + "' doesn't exist!") 

def format_detect(input_f = inp):
	if input_f.split('.')[-1] == 'mp3':
		print 'mp3 convertion to wav...'
		sound = AudioSegment.from_mp3(input_f)
		sound.export("temp.wav", format="wav"); input_f = "temp.wav"
	elif input_f.split('.')[-1] == 'wav':
		print 'already *.wav format'
	else:
		sys.exit("NOT AUDIO INPUT. (choose *.mp3 or *.wav)")

def convertion_to_text(input_f = inp, output_f = out):
	chunk_size = 1024
	with open(input_f, 'rb') as f, open(output_f+"_"+str(os.path.getsize(input_f)), 'w') as fout:
	    while True:
	        data = f.read(chunk_size)
	        if not data:
	            break
	        hexa = binascii.hexlify(data)
	        hexa_string = hexa.decode('ascii')
	        fout.write(hexa_string) 
	    print("converted to text in '" + output_f + "'")

read_file()
format_detect()
convertion_to_text(inp, out)

# os.remove("temp.wav") 

# with open("temp.wav") as f, open(out+"_"+str(frames), 'wb') as fout:
# 	for line in f:
# 		l = binascii.hexlify(''.join(line.split()))
# 		fout.write(l) 
# 	print("converted to text in '" + out + "'")