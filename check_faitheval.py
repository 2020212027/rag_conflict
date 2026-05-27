import pyarrow.parquet as pq
import json

path = 'd:/pythonProject/FaithEval_inconsistent/data/test-00000-of-00001.parquet'
table = pq.read_table(path)

print(f"Num rows: {table.num_rows}")
print(f"Columns: {table.column_names}")
print()

# First sample
for col in table.column_names:
    val = str(table.column(col)[0].as_py())
    if len(val) > 300:
        print(f"{col} (len={len(val)}): {val[:300]}...")
    else:
        print(f"{col}: {val}")
print()

# Context analysis
ctx_col = 'context' if 'context' in table.column_names else 'contexts'
ctx = str(table.column(ctx_col)[0].as_py())
print(f"--- Context analysis ---")
print(f"Has [Document: {'[Document' in ctx}")
print(f"Has 'Document 1': {'Document 1' in ctx}")
print(f"Has numbered [1]: {'[1]' in ctx}")
print(f"Newlines: {ctx.count(chr(10))}")
print()

# Sample 50
if table.num_rows > 50:
    ctx50 = str(table.column(ctx_col)[50].as_py())
    print(f"--- Sample 50 context[:300] ---")
    print(ctx50[:300])
    print()

# Category distribution
if 'category' in table.column_names:
    cats = [str(table.column('category')[i].as_py()) for i in range(table.num_rows)]
    from collections import Counter
    print(f"Categories: {dict(Counter(cats))}")
elif 'type' in table.column_names:
    types = [str(table.column('type')[i].as_py()) for i in range(table.num_rows)]
    from collections import Counter
    print(f"Types: {dict(Counter(types))}")
