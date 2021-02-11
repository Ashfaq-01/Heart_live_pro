import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.label import Label


class Open_screen(Screen):
    age = ObjectProperty(None)
    sex = ObjectProperty(None)
    cp = ObjectProperty(None)
    trestbps = ObjectProperty(None)
    chol = ObjectProperty(None)
    fbs = ObjectProperty(None)
    restecg = ObjectProperty(None)
    thalach = ObjectProperty(None)
    exang = ObjectProperty(None)
    oldpeak = ObjectProperty(None)
    pred = ObjectProperty(None)

    def predict(self):
        model = logistic()
        lst = [int(self.age.text), int(self.sex.text), int(self.cp.text), int(self.trestbps.text), int(self.chol.text),
               int(self.fbs.text),
               int(self.restecg.text), int(self.thalach.text), int(self.exang.text), float(self.oldpeak.text), 0, 1, 0]
        array = np.array(lst)
        array = array.reshape(1, -1)
        self.pred = model.predict(array)
        print(self.pred)
        print(model.predict_proba(array))
        if self.pred == 1:
            popup_1()

        else:
            popup()


class WindowManager(ScreenManager):
    pass


def popup():
    pop = Popup(title="Low risk",
                content=Label(text="You are at a low risk of getting a heart ailment"),
                size_hint=(None, None), size=(400, 400))

    pop.open()


def popup_1():
    pop = Popup(title="High risk", content=Label(text="You are at a high risk of developing a heart ailment"),
                size_hint=(None, None), size=(400, 400))

    pop.open()


def logistic():
    data = pd.read_csv(r"C:\Users\Ashfaaq ahmed\Downloads\heart.csv")
    y = data.iloc[:, 13]
    x = data.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = LogisticRegression(max_iter=1000)
    model = model.fit(x_train, y_train)
    model.predict(x_test)
    return model


kv = Builder.load_file("my.kv")
sm = WindowManager()
screens = [Open_screen(name="Open_screen")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "Open_screen"


class MyMainApp(App):

    def build(self):
        return sm


if __name__ == "__main__":
    MyMainApp().run()
