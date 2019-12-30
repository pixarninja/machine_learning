from my_lslr import my_lslr
from my_ridge import my_ridge
from lslr import lslr
from ridge import ridge
from lasso import lasso
from elastic import elastic
import utilities

# Load dataset.
normalized = True
dataset = utilities.import_CCPP(normalized)

fit,_ = my_lslr(dataset, 15000, 0.1)
utilities.stats(dataset, fit, 'My LSLR')

fit,_ = my_ridge(dataset, 15000, 0.1, 0.1)
utilities.stats(dataset, fit, 'My Ridge')

fit = lslr(dataset)
utilities.stats(dataset, fit, 'Library LSLR')

fit = ridge(dataset)
utilities.stats(dataset, fit, 'Library Ridge')

# Load dataset.
normalized = False
dataset = utilities.import_CCPP(normalized)

fit = elastic(dataset)
utilities.stats(dataset, fit, 'Library Elastic')

fit = lasso(dataset)
utilities.stats(dataset, fit, 'Library Lasso')
