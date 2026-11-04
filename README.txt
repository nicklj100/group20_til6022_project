================================================================================
Weather Impact on the Operations of Utrecht Centraal Station
TIL6022 - Python Programming
Group Project (Group 20)
================================================================================

1. PROJECT OVERVIEW
-------------------

This project analyses the operational impact of the heavy snowstorm that
occurred from January 6th to January 8th, 2026, on the Dutch railway network,
specifically focusing on Utrecht Centraal Station.

Primary Research Question:

"How did the snowstorm of 6 January 2026 affect the duration of train delays,
the number of cancellations, and punctuality at Utrecht Centraal, compared with
a typical day under normal weather conditions?"

Objectives:
- Establish a baseline of NS train operations and performance at Utrecht
  Centraal under normal weather conditions, including delays, cancellations,
  and punctuality.
- Evaluate NS operations at Utrecht Centraal on 6 January 2026 during the
  snowstorm.
- Compare baseline metrics against the snowstorm period to determine the
  overall operational impact.

2. AUTHORS (GROUP 20)
---------------------

- Savino Every
- Nicholas Johnson
- Gijs van Vliet
- Delano van den Bosch

Note: Student IDs are omitted in compliance with TIL6022 repository guidelines.

3. DATASETS & SOURCES
---------------------

KNMI Hourly Weather Data

Weather records from De Bilt weather station, located approximately 4 km
northeast of Utrecht Centraal. The dataset includes temperature, precipitation
amount and duration, maximum hourly precipitation, and weather codes related
to snowfall, icing, rain, and thunderstorms.

Source:
https://www.knmi.nl/nederland-nu/klimatologie/uurgegevens


Rijden de Treinen Open Data

Detailed records of train services at Utrecht Centraal, including
arrival and departure delays and partial or full cancellations.

Source:
https://www.rijdendetreinen.nl/open-data/treinarchief


4. REPOSITORY STRUCTURE
-----------------------

.
├── data/
│   ├── weather_selected.csv
│   │   # KNMI hourly weather records -> initialized in data_import.py
│   └── traindata_total.csv
│       # Rijden de Treinen train performance data -> initialized in data_import.py
│
├── background/
│   ├── data_import.py
│   │   # Import of data sources offline to create smaller datasources (redundant, for information)
│   ├── train_analysis.py
│   │   # Concatenated train_analysis notebook for implementation in project_final_report
│   ├── weather.ipynb
│   │   # Full data analysis on weather data
│   └── train_analysis.ipynb
│       # Full data analysis on train data
│
├── project_final_report.ipynb
│   # Main Jupyter notebook containing the narrative and analysis
│
└── README.txt
    # Project documentation and reproduction guide


5. REQUIREMENTS & DEPENDENCIES
------------------------------

- Python 3.10+
- Jupyter Notebook or JupyterLab

Required Python packages:

- pandas
- numpy
- matplotlib
- seaborn

Install the required dependencies using pip:

    pip install pandas numpy matplotlib seaborn jupyter


6. REPRODUCTION & EXECUTION PIPELINE
------------------------------------

The complete project repository is available on GitHub:

https://github.com/nicklj100/group20_til6022_project

Step 1: Clone the repository

    git clone https://github.com/nicklj100/group20_til6022_project.git

Step 2: Navigate to the project directory

    cd group20_til6022_project

Step 3: Install the required dependencies

    pip install pandas numpy matplotlib seaborn jupyter

Step 4: Ensure that the required datasets are located in the data/ directory

    data/
!    ├── xxxxx.csv
!    ├── xxxxx.csv
    └── traindata_total.csv

Step 5: Start Jupyter Notebook

Step 6: Open and run the main analysis notebook

    project_final_report.ipynb

Run all cells in order to reproduce the complete data pipeline, analysis,
statistical results, and visualizations.

The analysis pipeline consists of the following main stages:

1. Loading and cleaning the KNMI weather data and Rijden de Treinen train
   service data.
2. Filtering the train data for relevant services and the Utrecht Centraal
   study period.
3. Processing arrival and departure delays and identifying cancelled services.
4. Calculating operational performance indicators, including delay duration,
   cancellations, and punctuality.
5. Establishing baseline performance under typical weather conditions.
6. Identifying and analysing operations during the snowstorm period.
7. Comparing snowstorm performance with the baseline.
8. Generating visualizations and conclusions regarding the operational impact
   of the snowstorm.

================================================================================