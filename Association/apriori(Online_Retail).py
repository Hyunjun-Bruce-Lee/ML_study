# apriori를 이용한 장바구니 데이터 (market basket) 분석 예시 (3)
# 참고 사이트  https://pbpython.com/market-basket-analysis.html
# -------------------------------------------------------------
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Online Retail 데이터를 읽어온다.
# 데이터 출처 : https://archive.ics.uci.edu/ml/datasets/online+retail
basket = pd.read_csv('dataset/OnlineRetail.csv')

# 데이터가 커서 France 데이터만 읽어온다.
basket = basket[basket['Country'] == "France"]

# Description의 맨 앞과 뒤에 space, newline 문자가 있으면 제거한다.
basket['Description'] = basket['Description'].str.strip()

# InvoiceNo 자리에 nan이 있으면 이 행을 제거한다
basket.dropna(axis=0, subset=['InvoiceNo'], inplace=True)

# InvoiceNo 앞이나 뒤에 'C'가 붙어있는 경우가 있다. 이 행을 제거한다.
basket = basket[~basket['InvoiceNo'].str.contains('C')]

# InvoiceNo를 string 형태로 변환한다.
basket['InvoiceNo'] = basket['InvoiceNo'].astype('str')

# 동일한 InvoiceNo를 그룹으로 묶는다.
basket = basket.groupby(['InvoiceNo', 'Description'])['Quantity'].sum()
basket = basket.unstack()
basket = basket.fillna(0)
basket = basket.reset_index()
basket = basket.set_index('InvoiceNo')

# Quantity가 1보다 큰 것은 모두 1로 바꾼다.
basket[basket <= 0] = 0
basket[basket >= 1] = 1

# apriori 알고리즘을 적용한다.
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# association rule을 찾는다.
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# lift가 큰 것부터 나오도록 sort한다. descending.
rules = rules.sort_values(by=['lift'], axis=0, ascending=False)

# 'antecedents','consequents' 행이 frozenset() 형으로 돼있어서 보기 불편하므로
# 일반 tuple형태로 변환한다.
cols = ['antecedents','consequents']
rules[cols] = rules[cols].applymap(lambda x: tuple(x))

# 결과를 파일에 저장한다.
rules.to_csv('dataset/tmp.csv')

