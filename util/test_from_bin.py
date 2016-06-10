import sys, os.path
import unittest
from from_bin import read_file
from from_bin import format_detect
from from_bin import convertion_to_text
from StringIO import StringIO
import re
import subprocess

def istext(path):
    return (re.search(r':.* text',
                      subprocess.Popen(["file", '-L', path], 
                                       stdout=subprocess.PIPE).stdout.read())
            is not None)

class FromBinTestCase(unittest.TestCase):
	""" Tests for 'from_bin.py' """
	def setUp(self):
		self.held, sys.stdout = sys.stdout, StringIO()

	def test_not_existing_file(self):
		""" Is the input path exists? """
		file_path = "nothing.mp3"
		with self.assertRaises(SystemExit) as cm:
		    read_file(file_path)

		self.assertEqual(cm.exception.code, "Input file 'nothing.mp3' doesn't exist!")
	
	"""Detection of the format"""
	def test_wav_input(self):
		""" Is any additional binary covertion needed for *.wav? """
		file_path = "austspring.wav"
		format_detect(file_path)
		self.assertEqual(sys.stdout.getvalue(),'already *.wav format\n')

	def test_mp3_input(self):
		""" Is covertion from *.mp3 into *.wav needed? """
		file_path = "austspring.mp3"
		format_detect(file_path)
		self.assertEqual(sys.stdout.getvalue(),'mp5 convertion to wav...\n')

	def test_mp3_convertion(self):
		""" Is the covertion successful from *.mp3 into *.wav? """
		if os.path.isfile("temp.wav"):
			os.remove("temp.wav") 
		file_path = "austspring.mp3"
		format_detect(file_path)
		self.assertTrue(os.path.isfile("temp.wav"))
		if os.path.isfile("temp.wav"):
			os.remove("temp.wav") 

	def test_not_audio_input(self):
		""" Is the input file an audio or not? """
		file_path = 'misc.lua'

		with self.assertRaises(SystemExit) as cm:
		    format_detect(file_path)

		self.assertEqual(cm.exception.code, "NOT AUDIO INPUT. (choose *.mp3 or *.wav)")

	def test_convert_to_text(self):
		""" Check if the conversion from bin into text is successful """
		file_path = "austspring.wav"
		convertion_to_text(file_path, "test_output.txt")
		self.assertFalse(istext(file_path))
		self.assertTrue(istext("test_output.txt"+"_"+str(os.path.getsize(file_path))))


if __name__ == '__main__':
    unittest.main()
