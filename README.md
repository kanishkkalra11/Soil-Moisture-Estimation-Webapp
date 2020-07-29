# Soil-Moisture-Estimation-Webapp
This is a django based app. To host on the local machine, run the following on command line:
$python manage.py runserver$


users folder contains the utilities for adding new users, login, logout, sign in, etc.

app folder contains the utilities of actual moisture estimation.

The actual image processing is done using count_moisture_content.py and img_tools.py. This is taken from:
_V.K. Gadi, D. Alybaev, P. Raj, A. Garg, G. Mei, S. Sreedeep,
and L. Sahoo, “A Novel Python Program to Automate Soil
Colour Analysis and Interpret Surface Moisture Content,”
International Journal of Geosynthetics and Ground
Engineering, Vol. 6, p. 21, May. 2020_
