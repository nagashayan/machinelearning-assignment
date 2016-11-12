from sknn.mlp import Classifier,Layer
import pcaTest as pc
import dataProcessor as dp
import numpy as np

nn = Classifier(
    layers=[
        Layer("Rectifier", units=100),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)
x = nn.fit(pc.transformed_trainData1_norm.values, dp.TrainData1_labels.values)
