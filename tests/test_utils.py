"""Tests for frequency inference."""

import pandas as pd
import pytest

from alphalytics.utils import _infer_periods_per_year


class TestInferPeriodsPerYear:
    def test_biweekly(self):
        index = pd.date_range("2020-01-03", periods=20, freq="2W")
        assert _infer_periods_per_year(index) == 26

    def test_monthly(self, monthly_index):
        assert _infer_periods_per_year(monthly_index) == 12

    def test_quarterly(self):
        index = pd.date_range("2015-03-31", periods=20, freq="QE")
        assert _infer_periods_per_year(index) == 4

    def test_business_days(self, daily_index):
        assert _infer_periods_per_year(daily_index) == 252

    def test_seven_day_data(self):
        index = pd.date_range("2020-01-01", periods=400, freq="D")
        assert _infer_periods_per_year(index) == 365

    def test_sunday_to_thursday_market(self):
        index = pd.bdate_range("2020-01-01", periods=400, freq="C",
                               weekmask="Sun Mon Tue Wed Thu")
        assert _infer_periods_per_year(index) == 252

    def test_hourly_raises(self):
        index = pd.date_range("2020-01-01", periods=48, freq="h")
        with pytest.raises(ValueError):
            _infer_periods_per_year(index)

    def test_single_date_raises(self):
        index = pd.DatetimeIndex([pd.Timestamp("2020-01-31")])
        with pytest.raises(ValueError):
            _infer_periods_per_year(index)
