import pandas as pd
from sklearn.linear_model import LogisticRegression
files = ["E0.csv", "E0-2.csv", "E0-3.csv", "E0-4.csv"]
cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
df = pd.concat([pd.read_csv(f, usecols=cols) for f in files])

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].head())
print(df["FTR"].value_counts(normalize=True))
print("-"*100)
df["MatchID"] = df.index

home = df[["MatchID", "Date", "HomeTeam", "FTHG", "FTAG"]]
home.columns = ["MatchID", "Date", "Team", "GF", "GA"]
away = df[["MatchID", "Date", "AwayTeam", "FTAG", "FTHG"]]
away.columns = ["MatchID", "Date", "Team", "GF", "GA"]

long = pd.concat([home, away]).sort_values(["Team", "Date"]).reset_index(drop=True)

g = long.groupby("Team")
long["GFS"] = g["GF"].transform(lambda s: s.shift(1).rolling(5).mean())
long["GAS"] = g["GA"].transform(lambda s: s.shift(1).rolling(5).mean())

feat = long[["MatchID", "Team", "GFS", "GAS"]]

df = df.merge(feat.rename(columns={"Team":"HomeTeam", "GFS":"H_GFS", "GAS":"H_GAS"}), on=["MatchID", "HomeTeam"])

df = df.merge(feat.rename(columns={"Team":"AwayTeam", "GFS":"A_GFS", "GAS":"A_GAS"}), on=["MatchID", "AwayTeam"])

df = df.dropna(subset=["H_GFS", "H_GAS", "A_GFS", "A_GAS"]).reset_index(drop=True)
print(df.shape)
print(df[["Date", "HomeTeam", "AwayTeam", "H_GFS", "H_GAS", "A_GFS", "A_GAS"]].head())

features = ["H_GFS", "H_GAS", "A_GFS", "A_GAS"]
train = df[df["Date"] < "2024-08-01"]
test = df[df["Date"] >= "2024-08-01"]
print("-"*100)
print("train rows:", len(train), "| test rows:", len(test))
print("baseline on test (always guess home win):", (test["FTR"] == "H").mean())

model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["FTR"])

prediction = model.predict(test[features])

print("model accuracy:", (prediction == test["FTR"]).mean())
print(pd.Series(prediction).value_counts())
print("-"*100)
print(pd.crosstab(test["FTR"], prediction, rownames=["actual"], colnames=["predicted"]))
print("-"*100)

probabilities = model.predict_proba(test[features])
show = test[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
show["predicted"] = prediction
show["P(home win)"] = probabilities[:, list(model.classes_).index("H")].round(2)
print(show.head(10))
