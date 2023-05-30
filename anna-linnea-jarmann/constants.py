import pandas as pd

REDUCED_COVARIATES = ["acwr",
                      "sleep_duration",
                      "readiness",
                      "fatigue"]

REG_COVARIATES = ["acwr",
                  "fatigue",
                  "mood",
                  "readiness",
                  "sleep_duration",
                  "sleep_quality",
                  "soreness",
                  "stress"]

ALL_COVARIATES = ["daily_load",
                  "atl",
                  "weekly_load",
                  "monotony",
                  "strain",
                  "acwr",
                  "ctl28",
                  "ctl42",
                  "fatigue",
                  "mood",
                  "readiness",
                  "sleep_duration",
                  "sleep_quality",
                  "soreness",
                  "stress"]

DURATIONS_EXAMPLE = pd.DataFrame({
    "duration": [8, 6, 25, 10, 4, 10, 45, 5, 12, 18],
    "event": [True, True, False, True, True, True, False, True, True, True]
})

DURATIONS_RECURRENT_EXAMPLE = pd.DataFrame({
    "duration": [8, 6, 10, 25, 4, 10, 45, 5, 12, 18],
    "event": [True, True, True, False, True, True, False, True, True, True]
}, index=["0", "0", "0", "1", "2", "2", "3", "4", "4", "4"])