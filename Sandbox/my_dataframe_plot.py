#!/usr/bin/env python3

import pandas
import random
from matplotlib import pyplot as plt

if __name__ == "__main__":
    data = []
    for distance in [10, 30, 50]:
        for pfail in [0, 0.1]:
            for num_node in range(10):
                data.append({
                    'distance': distance,
                    'pfail': pfail,
                    'num_node': num_node,
                    'attempts': random.random(),
                    'successful': random.randint(0, 10),
                    'success_prob': random.randint(-20, 20)
                    })

    df = pandas.DataFrame(data)

    metrics = ['attempts', 'successful', 'success_prob']

    for metric in metrics:
        _, ax = plt.subplots()
        
        for distance in [10, 30, 50]:
            for pfail in [0, 0.1]:
                df.loc[(df['distance'] == distance) & (df['pfail'] == pfail)].\
                    plot(x='num_node', y=metric, label=f"{distance} km, p = {pfail}", ax=ax)

        plt.xlabel("x-label")
        plt.ylabel(metric)
        plt.title("Plot title")
        plt.show(block=(metric == metrics[-1]))
