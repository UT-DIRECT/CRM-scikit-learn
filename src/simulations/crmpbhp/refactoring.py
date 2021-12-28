import cProfile
import pstats
from pstats import SortKey

import numpy as np

from crmp import CrmpBHP
from sklearn.model_selection import train_test_split

from src.data.read_crmp import injectors, producers
from src.helpers.features import production_rate_dataset
from src.simulations import number_of_producers


def fit_all_producers():
    for i in range(number_of_producers):
        producer = producers[i]
        X, y = production_rate_dataset(producer, *injectors)
        q = X[:, 0].reshape(150, 1)
        delta_p = np.zeros((len(X) , 1))
        inj1 = X[:, 1].reshape(150, 1)
        inj2 = X[:, 2].reshape(150, 1)
        X = np.hstack((q, delta_p, inj1, inj2))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        crmp = CrmpBHP(q0=producer[0])
        crmp = crmp.fit(X_train, y_train)
        print('Producer {}'.format(i + 1))
        print('Tau: {}'.format(crmp.tau_))
        print('Gains: {}'.format(crmp.gains_))
        print()


fit_all_producers()
# cProfile.run('fit_all_producers()', 'output.dat')
#
# with open('output_time.txt', 'w') as f:
#     p = pstats.Stats('output.dat', stream=f)
#     p.sort_stats('time').print_stats()
#
# with open('output_calls.txt', 'w') as f:
#     p = pstats.Stats('output.dat', stream=f)
#     p.sort_stats('calls').print_stats()
