import pandas as pd
import numpy as np
import sys

from IPython.display import display, HTML
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format
import pandas.core.algorithms as algos


from pandas.core.dtypes.common import (
    is_integer,
    is_scalar,
    is_categorical_dtype,
    is_datetime64_dtype,
    is_timedelta64_dtype,
    _ensure_int64)

from pandas.core.dtypes.missing import isnull

from pandas import (to_timedelta, to_datetime,
                    Categorical, Timestamp, Timedelta,
                    Series, Interval, IntervalIndex)



pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 200

pd.set_option('display.width', 1000)
from IPython.display import display, HTML
import pandas.core.algorithms as algos
from pandas.api.types import is_numeric_dtype
from pandas.api.types import CategoricalDtype

sys.path.append("/Users/hemin/AnacondaProjects/Gitfolder/python_analytic_functions/")

from score_dist_fn import bin_cut

