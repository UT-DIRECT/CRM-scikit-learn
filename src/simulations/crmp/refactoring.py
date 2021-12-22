import cProfile
import pstats
from pstats import SortKey

from sklearn.model_selection import train_test_split
from crmp import CRMP

from src.data.read_crmp import injectors, producers
from src.helpers.features import production_rate_dataset
from src.simulations import number_of_producers


def fit_all_producers():
    for i in range(number_of_producers):
        producer = producers[i]
        X, y = production_rate_dataset(producer, *injectors)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.5, shuffle=False
        )
        crmp = CRMP(q0=producer[0])
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
