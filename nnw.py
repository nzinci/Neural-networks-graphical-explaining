import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QDockWidget, QWidget, QGridLayout, QSlider, QLabel
from PyQt5.QtCore import Qt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_qt4agg

class MainWindow(QMainWindow):
    x = np.arange(1, 100, 0.001)
    delta = 1
    lim = 13
    A1 = np.random.rand(2,2)
    A2 = np.random.rand(2,2)
    b1 = np.array([0, 0])
    b2 = np.array([0, 0])

    def __init__(self):
        QMainWindow.__init__(self)

        self.figure  = plt.figure()
        self.drawing = self.figure.add_subplot(122)
        self.another = self.figure.add_subplot(121)
        self.canvas  = matplotlib.backends.backend_qt4agg.FigureCanvasQTAgg(self.figure)

        self.setCentralWidget(self.canvas)

        dock = QDockWidget("Values")
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        sliders = QWidget()
        sliders_grid = QGridLayout(sliders)

        def add_slider(foo, col, row):
            sld = QSlider(Qt.Horizontal, sliders)
            sld.setMinimum(-10)
            sld.setMaximum(10)
            sld.setFocusPolicy(Qt.NoFocus)
            sld.valueChanged[int].connect(foo)
            sld.valueChanged.connect(self.plot)
            sliders_grid.addWidget(sld, row, col)

        add_slider(foo=self.a00, col=0, row=0)
        add_slider(foo=self.a01, col=0, row=1)
        add_slider(foo=self.a10, col=0, row=2)
        add_slider(foo=self.a11, col=0, row=3)
        add_slider(foo=self.c00, col=1, row=0)
        add_slider(foo=self.c01, col=1, row=1)
        add_slider(foo=self.c10, col=1, row=2)
        add_slider(foo=self.c11, col=1, row=3)
        add_slider(foo=self.b10, col=2, row=0)
        add_slider(foo=self.b11, col=2, row=1)
        add_slider(foo=self.b20, col=3, row=0)
        add_slider(foo=self.b21, col=3, row=1)
        

        dock.setWidget(sliders)

        self.plot()
    
    def a00(self, val):
        self.A1[0][0] = val

    def a01(self, val):
        self.A1[0][1] = val

    def a10(self, val):
        self.A1[1][0] = val

    def a11(self, val):
        self.A1[1][1] = val

    def c00(self, val):
        self.A2[0][0] = val

    def c01(self, val):
        self.A2[0][1] = val

    def c10(self, val):
        self.A2[1][0] = val

    def c11(self, val):
        self.A2[1][1] = val

    def b10(self, val):
        self.b1[0] = val    

    def b11(self, val):
        self.b1[1] = val

    def b20(self, val):
        self.b2[0] = val

    def b21(self, val):
        self.b2[1] = val

    def sigm(self, x):
        return 1/(1+np.exp(-x))


    def neural_net(self, x, A1, A2, b1, b2, ind):
        def layer(x, A, b):
            return self.sigm(A.dot(x) + b)
        y1 = layer(x, A1, b1)
        y2 = layer(y1, A2, b2)
        return y2[ind]

    datx = np.arange(-13, 13, 1)
    daty = np.arange(-13, 13, 1)

    X, Y = np.meshgrid(np.arange(-13, 13, 1), np.arange(-13, 13, 1))

    #Z = X + Y
    def make_z(self, ind):
        Z = []
        for i in self.daty:
            ls = []
            for j in self.datx:
                ls.append(self.neural_net(np.array([j, i]), self.A1, self.A2, self.b1, self.b2, ind))
            Z.append(ls)
        return Z

    def plot(self):
        self.drawing.hold(False)
        self.another.contourf(self.X, self.Y, self.make_z(1))
        self.drawing.contourf(self.X, self.Y, self.make_z(0))
        self.drawing.set_ylim(-10, 10)
        self.canvas.draw()


if __name__ == "__main__":
  app = QApplication(sys.argv)
  main = MainWindow()
  main.show()
  sys.exit(app.exec_())
