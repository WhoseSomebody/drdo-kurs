#!/usr/bin/env python
# coding=cp1251
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.button import Button
from subprocess import call
import os
 
from kivy.config import Config
Config.set('graphics', 'width', '500')
Config.set('graphics', 'height', '300')
from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)


class LoadTargetDialog(FloatLayout):
    load_audio = ObjectProperty(None)
    cancel = ObjectProperty(None)
    working_dir = os.path.dirname(os.path.realpath(__file__))

class PopupMessage(FloatLayout):
    cancel = ObjectProperty(None)

class Root(FloatLayout):
    path = ''

    def dismiss_popup(self):
        self._popup.dismiss()
    def _dismiss_popup(self):
        self.popup.dismiss()

    def load_audio(self, path, filename):
        self.dismiss_popup()
        self.path = str(filename[0])
        self.popup_mess('Successfully Uploaded "' + str(filename[0]) + '"')
        self.dismiss_popup()

    def close_all(self):
        App.get_running_app().stop()

    def popup_mess(self, smth):
        content = PopupMessage(cancel=self._dismiss_popup)
        self.popup = Popup(title=smth, content=content, size_hint=(0.7, 0.4))
        self.popup.open()

    def start_work(self):
        content = LoadTargetDialog(
            load_audio=self.load_audio, cancel=self.dismiss_popup)
        self._popup = Popup(title="Choose audiofile (*.mp3 or *.wav)", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def start_process(self):
        if len(self.path) > 0:
	    print self.path
            call(["./appstart.sh", "-i", self.path])
            self.new_but()
        

    def new_but(self):
        self.popup_mess("DONE! Check the 'result' folder.")
        


class Editor(App):
    icon = 'note.png'
    title = 'EXTRACTOR v1.0 (beta)'
    pass

Factory.register('Root', cls=Root)
Factory.register('LoadTargetDialog', cls=LoadTargetDialog)

if __name__ == '__main__':
    Editor().run()
