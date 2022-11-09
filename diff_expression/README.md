# Gene differential expression
This project will help you analyze genes differential expression.
You can use diff_expression.py to run it locally in terminal or use My_homework_lecture_5.ipynb to run it in interactive mode.

The script diff_expression.py was run on Ubuntu 21.04 with Python 3.9.7.

1. Clone this repo. 
`git clone https://github.com/NatashaKhotkina/Stat_BI_2022`
2. Type 
`cd Stat_BI_2022/diff_expression`

You can either use jupyter notebook or the script. 
To use jupyter notebook open it in google-collab, for example.
To use script follow these steps:


3. Create a virtual environment. 
An easy way to do this is to run 
`conda create -n env python=3.9.7`
where 'env' is the name of your virtual environment (you can use any name you like).
4. Don't forget to activate the virtual environment
`conda activate env`
5. Then install all the requirments
`pip install -r requirements.txt`. 
6. Then run the code typing 
`diff_expression.py first_cell_type_expressions_path second_cell_type_expressions_path results_table_path`.

Or

`diff_expression.py first_cell_type_expressions_path second_cell_type_expressions_path results_table_path -cm correction_method`.

Where the **first argument** is the path to the first data table, the **second argument** is the path to the second data table,
the **third argument** is the path to the file where you want to save results (it'll be created or rewritten).
The **fourth argument** is optional (default = None) and takes the name of correction method for multiple comparison you want 
to use. (You can use: *bonferroni, sidak, holm-sidak, holm, simes-hochberg, hommel, fdr_bh, fdr_by, fdr_tsbh, fdr_tsbky*)

## Output:
The result table contains following columns:

- 'ci_test_results' with results of confident interval test: True if CI are not intersected, False if CI are intersected;

- 'z_test_results' with results of z test: True if H1 is correct, False if H0 is correct;

- 'z_test_p_values' with p values of z test;

- 'mean_diff' with difference of mean expression of the gene in two types of cell.
## Examples:
`python diff_expression.py b_cells_expression_data.csv nk_cells_expression_data.csv results.csv`

`python diff_expression.py b_cells_expression_data.csv nk_cells_expression_data.csv results.csv -cm bonferroni
`

Congratulations, you are awsome!   