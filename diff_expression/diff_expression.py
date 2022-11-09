import pandas as pd
import numpy as np
import argparse
import scipy.stats as st

from statsmodels.stats.weightstats import ztest


parser = argparse.ArgumentParser()
parser.add_argument('first_cell_type_expressions_path', help='Pathway to table with expressions of first cell type')
parser.add_argument('second_cell_type_expressions_path', help='Pathway to table with expressions of second cell type')
parser.add_argument('save_results_table', help='Pathway to table where you want to save the results')
parser.add_argument('--correction_method', '-cm', const=None, help='Correction method for multiple comparison. You can use: '
                                                                   'bonferroni, sidak, holm-sidak, holm, simes-hochberg, '
                                                                   'hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky')

args = parser.parse_args()
first_table = pd.read_csv(f"{args.first_cell_type_expressions_path}", index_col=0)
second_table = pd.read_csv(f"{args.second_cell_type_expressions_path}", index_col=0)


def check_intervals_intersect(first_ci, second_ci):
    are_intersect = (first_ci[0] < second_ci[0] < first_ci[1]) or (second_ci[0] <
    first_ci[0] < second_ci[1])

    return are_intersect


def check_dge_with_ci(first_table, second_table):
    ci_test_results = []
    for gene_name in first_table.columns[:-1]:
      b_ci = st.t.interval(alpha=0.95,
                  df=len(first_table[gene_name]) - 1,
                  loc=np.mean(first_table[gene_name]),
                  scale=st.sem(first_table[gene_name]))

      nk_ci = st.t.interval(alpha=0.95,
                  df=len(second_table[gene_name]) - 1,
                  loc=np.mean(second_table[gene_name]),
                  scale=st.sem(second_table[gene_name]))

      ci_test_results.append(not(check_intervals_intersect(b_ci, nk_ci)))

    return ci_test_results


def check_dge_with_ztest(first_table, second_table):
    z_test_p_values = []
    for gene_name in first_table.columns[:-1]:
        p_value = ztest(first_table[gene_name], second_table[gene_name])[1]
        z_test_p_values.append(p_value)

    return z_test_p_values


def get_mean_diff(first_table, second_table):
    mean_diff = []
    for gene_name in first_table.columns[:-1]:
      first_mean = np.mean(first_table[gene_name])
      second_mean = np.mean(second_table[gene_name])
      mean_diff.append(first_mean - second_mean)

    return mean_diff


ci_test_results = check_dge_with_ci(first_table, second_table)
z_test_p_values = check_dge_with_ztest(first_table, second_table)
mean_diff = get_mean_diff(first_table, second_table)


if args.correction_method is not None:
    from statsmodels.stats.multitest import multipletests
    z_test_p_values = multipletests(pvals=z_test_p_values, method=args.correction_method)[1]

z_test_p_values = np.array(z_test_p_values)
z_test_results = z_test_p_values < 0.05


results = {
    "ci_test_results": ci_test_results,
    "z_test_results": z_test_results,
    "z_test_p_values": z_test_p_values,
    "mean_diff": mean_diff
}

indecies = first_table.columns[:-1]
results = pd.DataFrame(results, index=indecies)
results.to_csv(f"{args.save_results_table}")