import pstats


def main():
    with open('stats_out.txt', 'w') as stream:
        s = pstats.Stats('stats.txt', stream=stream)
        s.sort_stats('cumulative').print_stats()


if __name__ == "__main__":
    main()
