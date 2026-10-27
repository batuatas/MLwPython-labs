# Machine Learning with Python: laptop labs

Lab notebooks and data for the course *Machine Learning with Python* (6012B0419Y), Minor Amsterdam Data Science and Artificial Intelligence, University of Amsterdam, Fall 2026.

Lecturer: Ş. İlker Birbil · Laptop labs: Batuhan Ataş (b.atas@uva.nl)

## How to use

Open a notebook in Google Colab with the links below, then choose **File → Save a copy in Drive** and work in your copy. The Laptop lab guide on Canvas (Week 1) explains the details, including how to work on your own laptop instead.

| Notebook | When | Open |
|---|---|---|
| Setup check | before the first lab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab00_setup_check.ipynb) |
| Lab 1: Visualization | Thursday, October 29 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab01.ipynb) |

## Data

| File | Content |
|---|---|
| `data/student_survey.csv` | synthetic student survey used in the lectures (300 students) |
| `data/student_survey_raw.csv` | the same survey with typical data-quality problems |
| `data/airbnb_listings.csv` | Airbnb listings in Amsterdam, 16 September 2026: 10,612 listings, 20 columns (Lab 1) |
| `data/airbnb_reviews_daily.csv` | number of reviews of Amsterdam listings per day, 2010 to 2026 (Lab 1) |

The notebooks read the data directly from this repository, so you do not need to download them.

## Credits

The Airbnb data come from [Inside Airbnb](https://insideairbnb.com) (Amsterdam, 16 September 2026) and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We kept a selection of the columns of the listings, without names, texts or links, and counted the reviews per day.
