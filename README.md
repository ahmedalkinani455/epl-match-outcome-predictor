# Premier League Match Outcome Model
Predicts home win / draw / away win for Premier League matches from recent team form.

## Data
Used data from football-data.co.uk, 2021/22 to 2024/25 Premier League seasons (1,520 matches; 1,440 were useable after requiring 5 prior games per team: 1,065 train, 375 test)

## Method
- Features: each team's average goals scored and conceded over its previous 5 matches
- Split by time: trained on 2021/22-2023/24 and tested on 2024/25
- Model: logistic regression (scikit-learn)

## Results
- Accuracy on the 2024/25 test season: 48.0%
- Baseline (always predict home win): 41.1%
- Correctly predicts 78% of home wins and 46% of away wins, but 0% if draws

## Limitations
- Tested on a single season (375 matches), so the 7-point edge over the baseline could partly be noise
- Only uses recent goals scored/conceded; ignores team quality, injuries, and home advantage by team
- Never predicts draws, which caps its accuracy

