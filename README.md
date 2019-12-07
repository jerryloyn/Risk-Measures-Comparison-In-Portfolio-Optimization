# Risk Measure Comparison In Portfolio Optimization

In this study, we compare the performance (Sharpe Ratio) of 3 risk minimization models: Minimax, Variance and Absolute Deviation. In-sample and Out-of-sample analysis are conducted using histoical HSI stocks data.

## Setup (Windows only)

#### 1. Create a virtual environment in a directory

``` command
python3 -m venv portfolioOptimization
```

#### 2. Activate the virtual environment

``` command
portfolioOptimization\Scripts\activate.bat
```

#### 3. Put `result.py`, `portfolioOptimization.py` and `requirements.txt` in the same directory

#### 4. Install required packages

``` command
pip install -r requirements.txt
```

#### 5. Run the code

``` command
python result.py
```

## References

* https://papers.nips.cc/paper/5714-robust-portfolio-optimization.pdf
* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1866885
* http://www.scielo.br/pdf/prod/v27/0103-6513-prod-0103-6513208816.pdf
* https://towardsdatascience.com/efficient-frontier-portfolio-optimisation-in-python-e7844051e7f
