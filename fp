# --------------------------------------------
# FP-Growth on Bread Basket dataset (Google Colab)
# --------------------------------------------

# Install dependencies
!pip install mlxtend pandas --quiet

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from google.colab import files

# Step 1: Upload the dataset
uploaded = files.upload()  # Upload bread basket.csv here

# Step 2: Read CSV
filename = list(uploaded.keys())[0]  # Take the uploaded filename
df_raw = pd.read_csv(filename)
df_raw.columns = [c.strip() for c in df_raw.columns]

# Step 3: Identify transaction & item columns (change if different in your file)
txn_col = "Transaction"   # Column with transaction IDs
item_col = "Item"         # Column with item names

# Step 4: Clean data
df = df_raw[[txn_col, item_col]].dropna()
df[item_col] = df[item_col].astype(str).str.strip()
df = df[df[item_col].str.upper() != "NONE"]

# Step 5: One-hot encoding for FP-Growth
basket = (
    df.drop_duplicates()
      .pivot_table(index=txn_col, columns=item_col, aggfunc=lambda x: 1, fill_value=0)
)
if isinstance(basket.columns, pd.MultiIndex):
    basket.columns = [col[-1] for col in basket.columns]
basket = basket.applymap(lambda x: 1 if x >= 1 else 0)

# Step 6: Run FP-Growth
min_support = 0.02  # 2% min support
frequent_itemsets = fpgrowth(basket, min_support=min_support, use_colnames=True)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets = frequent_itemsets.sort_values(["length", "support"], ascending=[True, False])

# Step 7: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
rules = rules.sort_values(["lift", "confidence"], ascending=False)

# Step 8: Save results
frequent_itemsets.to_csv("frequent_itemsets_fp_growth.csv", index=False)
rules.to_csv("association_rules_fp_growth.csv", index=False)

# Step 9: Download results
files.download("frequent_itemsets_fp_growth.csv")
files.download("association_rules_fp_growth.csv")

print("✅ Done! Files are ready for download.")
print(f"Frequent itemsets: {len(frequent_itemsets)} rows")
print(f"Association rules: {len(rules)} rows")

# Preview top results
print("\nTop 10 Frequent Itemsets:")
print(frequent_itemsets.head(10))
print("\nTop 10 Association Rules:")
print(rules.head(10))


using bread basket 

using brad basket
