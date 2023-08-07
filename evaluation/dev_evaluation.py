import sys
import numpy as np
import scipy.stats

def load_scores(input_file):
    with open(input_file) as f:
        return np.array([float(n) for n in f.read().split("\n") if n != ""])


def main(argv):
    _, input_file1, input_file2, output_file = argv

    res_scores = load_scores(input_file1)
    ref_scores = load_scores(input_file2)

    results = scipy.stats.kendalltau(res_scores, ref_scores)[0]

    with open(output_file, "w") as f:
        f.write("KENDALL: " + str(results))


# Run
if __name__ == '__main__':
    main(sys.argv)
