import pstats
from pstats import SortKey

p = pstats.Stats('l2cutprof')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)

