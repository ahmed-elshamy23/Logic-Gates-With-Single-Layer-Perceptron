##########################################################
##                Author: Ahmed Elshamy                 ##
##   3rd Year Electronics & Communication Engineering   ##
##                      Section 1                       ##
##########################################################
import os
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

eta = 0.1
trainingDataInputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
trainingDataOutputs = np.array([0, 0, 0, 1])
weights = np.array([rnd.random() for i in range(3)])
options = ["AND", "OR", "NAND", "NOR"]
optionsIndex = 0

print("A single-layer perceptron can model the following gates:\n1- AND\n2- OR\n3- NAND\n4- NOR")
choice = int(input("\nEnter the gate number: "))
match choice:
    case 1:
        trainingDataOutputs = np.array([0, 0, 0, 1])
    case 2:
        trainingDataOutputs = np.array([0, 1, 1, 1])
        optionsIndex = 1
    case 3:
        trainingDataOutputs = np.array([1, 1, 1, 0])
        optionsIndex = 2
    case 4:
        trainingDataOutputs = np.array([1, 0, 0, 0])
        optionsIndex = 3
    case _:
        print("\nYou didn\'t choose a gate, AND gate is chosen by default.")
print(f"\nInitial weights:\nw1 = {weights[0]}, w2 = {weights[1]}, b = {weights[2]}")

for i in range(10000):
    inducedLocalField = np.array([np.dot(
        trainingDataInputs[i], weights) for i in range(4)])
    output = np.array([1 if i > 0 else 0 for i in inducedLocalField])
    error = np.subtract(trainingDataOutputs, output)
    for i in range(4):
        if error[i]:
            weights += eta*error[i]*trainingDataInputs[i]

print(f"\nFinal weights for {options[optionsIndex]} gate:\nw1 = {weights[0]}, w2 = {weights[1]}, b = {weights[2]}")
print("\nDecision boundary was plotted successfully!\n")

m = -weights[0]/weights[1]
c = -weights[2]/weights[1]
x = np.linspace(-0.1, 1.1, 1000)
y = m*x + c
[plt.scatter(trainingDataInputs[i][0], trainingDataInputs[i][1], c='r' if trainingDataOutputs[i] == 0 else 'g') for i in range(4)]
plt.plot(x, y)
plt.grid()
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.title(f"Decision Boundary of {options[optionsIndex]} Gate")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
legendTitles = ["class 1 (s = 0)" if trainingDataOutputs[i] == 0 else "class 2 (s = 1)" for i in range(4)]
legendTitles.append("decision boundary")
plt.legend(legendTitles)
plt.show()
os.system("pause")
