small_df = df
small_df["int_rate"] = small_df["int_rate"].str.replace("%", "").astype(float) / 100
