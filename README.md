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
| `data/anscombe.csv` | Anscombe's quartet: four small datasets with the same summary statistics |
| `data/phds_arcades.csv` | math PhDs awarded and arcade revenue in the United States, 2000–2009 (Lecture 2) |
| `data/cbs_provinces.csv` | Statistics Netherlands (CBS) figures for the 12 provinces, 2003–2017 |

The notebooks read the data directly from this repository, so you do not need to download them.

**Sources.** The student survey is synthetic: it was generated for this course and describes no real students. `anscombe.csv` is from Anscombe (1973), *Graphs in statistical analysis*, The American Statistician 27(1). `phds_arcades.csv` holds the series used by Müller (2017) in Applied Machine Learning, after Tyler Vigen's *Spurious Correlations*. `cbs_provinces.csv` contains open data of Statistics Netherlands (CBS, StatLine), selected and reshaped for teaching; source: CBS. CBS does not endorse this course.
