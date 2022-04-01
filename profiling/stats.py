import pstats

with open('stats_out.txt', 'w') as stream:
    s = pstats.Stats('stats.txt', stream=stream)
    s.sort_stats('cumulative').print_stats()
