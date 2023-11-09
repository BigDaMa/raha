#!/usr/bin/env python3
# Generates synthetic datasets of online logins

# Schema: User, Timestamp, Country
# There are three users: 0, 1, and 2. User 1 always logs in from the same country. User 2 logs in from one country on weekdays, and another on weekends. User 3 logs in from various countries.

from countrydata import COUNTRY_DATA
import random
import utils

class User:
    def __init__(self, userid):
        self.uid = userid #"u" + str(
        self.has_outliers = True
        self.countries = [country[0] for country in utils.choose_n(2, COUNTRY_DATA)]

    def random_login(self):
        tsp = utils.random_timestamp()
        outlier = random.random() < OUTLIERS_RATE
        row = (self.uid, tsp, self.random_country(tsp, outlier))
        return outlier, row

    def random_country(self, tsp, outlier):
        pass

class Sedentary(User):
    def random_country(self, _, outlier):
        return self.countries[outlier]

class BusinessTraveler(User):
    def random_country(self, tsp, outlier):
        return self.countries[outlier ^ utils.isweekend(tsp)]

class FrequentFlyer(User):
    def __init__(self, userid):
        super().__init__(userid)
        self.has_outliers = False

    def random_country(self, tsp, outlier):
        return random.choice(self.countries)

OUTLIERS_RATE = 0.05

for uid, init in enumerate((Sedentary, BusinessTraveler, FrequentFlyer)):
    user = init(uid)
    utils.write_lines("logins{}".format(user.uid), 500, user.random_login, user.has_outliers)
