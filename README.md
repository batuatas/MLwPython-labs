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
| Lab 3: Linear classification and preprocessing | Thursday, November 12 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab03.ipynb) |
| Lab 4: Missing values, trees and forests | Thursday, November 19 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/lab04.ipynb) |
| Programming assignment 1 (template) | due Sunday, November 15 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/assignment1.ipynb) |
| Programming assignment 2 (template) | due Sunday, November 29 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/assignment2.ipynb) |
| Final group project (starter) | due Sunday, December 13 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/batuatas/MLwPython-labs/blob/main/project_starter.ipynb) |

## Data

| File | Content |
|---|---|
| `data/student_survey.csv` | synthetic student survey used in the lectures (300 students) |
| `data/student_survey_raw.csv` | the same survey with typical data-quality problems |
| `data/student-por.csv`, `data/student.txt` | Student Performance (Portuguese course, 649 students) and its codebook, for Programming assignment 1 |
| `data/adult.csv`, `data/adult.names` | Adult (1994 US census, a random sample of 10,000 of the 48,842 adults) and its codebook, for Programming assignment 2 |
| `data/cbs_kwb_2024.csv.gz`, `data/cbs_kwb_2024_codebook.csv` | CBS key figures for all municipalities, districts and neighborhoods of the Netherlands, 2024, with English column names, and the codebook, for the final group project |
| `data/rivm_health_2024.csv.gz`, `data/rivm_health_2024_codebook.csv` | RIVM health estimates for the same regions, 2024, adults, with English column names, and the codebook, for the final group project |
| `data/hotel_reviews.csv.gz` | 515,738 reviews of 1,492 hotels in six European cities, for the final group project |

The notebooks read these files directly from this repository, so you do not need to download them. Labs 1 and 2 download their data, the Airbnb listings of Amsterdam, directly from [Inside Airbnb](https://insideairbnb.com).

## Credits

The Airbnb data of Labs 1 and 2 come from [Inside Airbnb](https://insideairbnb.com) (Amsterdam, 16 September 2026) and are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The notebook downloads them from Inside Airbnb's website and keeps 20 of the 90 columns of the listings.

The data of Programming assignment 1 are the Student Performance data of P. Cortez (2008), UCI Machine Learning Repository, [doi:10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see also Cortez and Silva (2008), *Using data mining to predict secondary school student performance*, FUBUTEC 2008. The files are copied unchanged.

The data of Programming assignment 2 are a random sample of 10,000 adults (`random_state=0`) from the Adult data of B. Becker and R. Kohavi (1996), UCI Machine Learning Repository, [doi:10.24432/C5XW20](https://doi.org/10.24432/C5XW20), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see also Kohavi (1996), *Scaling up the accuracy of naive-Bayes classifiers: a decision-tree hybrid*, KDD-96. The training and test files of UCI are put together, with a header row; the values are as published, except that the final period of the test labels is removed. The codebook `adult.names` is copied unchanged.

The CBS data of the final group project are the *Kerncijfers wijken en buurten 2024* of Statistics Netherlands (CBS), version of October 2026, [cbs.nl](https://www.cbs.nl/nl-nl/maatwerk/2026/42/kerncijfers-wijken-en-buurten-2024), reused under the [CBS terms](https://www.cbs.nl/en-gb/about-us/website/copyright) (CC BY 4.0). The column names are translated into English, the region types (Gemeente, Wijk, Buurt) are written in English, and the CBS marker "." for an unknown value is left empty; the values are as published. The codebook lists the original CBS code of every column.

The health data of the final group project are the 2024 estimates for adults (18 or older) of RIVM, *Gezondheid per wijk en buurt; 2012/2016/2020/2022/2024 (indeling 2024)*, StatLine table 50150NED, based on the Gezondheidsmonitor Volwassenen en Ouderen of the GGDs, CBS and RIVM, [statline.rivm.nl](https://statline.rivm.nl), reused under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) with credit to RIVM. Only the estimates (not the 95% intervals) of 2024 are kept, the indicators not measured in 2024 are left out, and the column names are translated into English; the values are as published.

The hotel reviews are the *515K Hotel Reviews Data in Europe* of J. Liu (2017), [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe), released under CC0 (public domain). The file is compressed with gzip; its content is unchanged.
