import matplotlib.pyplot as plt

def vizualization(loss, accuracy):
    plt.plot(loss)
    plt.plot(accuracy)
    plt.show()