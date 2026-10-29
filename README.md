# Machine Learning with Python: laptop labs

Lab notebooks and data for the course *Machine Learning with Python* (6012B0419Y), Minor Amsterdam Data Science and Artificial Intelligence, University of Amsterdam, Fall 2026.

Lecturer: Ş. İlker Birbil · Laptop labs: Batuhan Ataş (b.atas@uva.nl)

## How to use

Open a notebook in Google Colab with the links below, then choose **File → Save a copy in Drive** and work in your copy. The Laptop lab guide on Canvas (Week 1) explains the details, including how to work on your own laptop instead.

| Notebook | When | Open |
|---|---|---|
| Setup check | before the first lab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab00_setup_check.ipynb) |
| Lab 1: Visualization | Thursday, October 29 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab01.ipynb) |
| Lab 2: Nearest neighbors and linear regression | Thursday, November 5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab02.ipynb) |
| Programming assignment 1 (template) | due Sunday, November 15 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/assignment1.ipynb) |

## Data

| File | Content |
|---|---|
| `data/student_survey.csv` | synthetic student survey used in the lectures (300 students) |
| `data/student_survey_raw.csv` | the same survey with typical data-quality problems |
| `data/student-por.csv`, `data/student.txt` | Student Performance (Portuguese course, 649 students) and its codebook, for Programming assignment 1 |

The notebooks read these files directly from this repository, so you do not need to download them. Labs 1 and 2 download their data, the Airbnb listings of Amsterdam, directly from [Inside Airbnb](https://insideairbnb.com).

## Credits

The Airbnb data of Labs 1 and 2 come from [Inside Airbnb](https://insideairbnb.com) (Amsterdam, 16 September 2026) and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The notebook downloads them from Inside Airbnb's website and keeps 20 of the 90 columns of the listings.

The data of Programming assignment 1 are the Student Performance data of P. Cortez (2008), UCI Machine Learning Repository, [doi:10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see also Cortez and Silva (2008), *Using data mining to predict secondary school student performance*, FUBUTEC 2008. The files are copied unchanged.
