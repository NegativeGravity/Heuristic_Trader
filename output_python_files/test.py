s = compute_feature(df)
assert isinstance(s, pd.Series)
assert s.name == FEATURE_CODE
assert s.index.equals(df.index)
assert len(s) == len(df)
print("OK:", FEATURE_CODE, len(s))