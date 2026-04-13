import pandas as pd
import os
import pathlib
from pathlib import Path
import random
import sys
import re
import math
import numpy as np

class ExcelParser:
    def __init__(self, input_filename=None):
        self.input_filename = Path(input_filename)
        self.blocker_info = []
        self.output_info = []
        self.output_folder = self.input_filename.parent / "output"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def read_excel_to_dfs(self):
        try:
            filename = self.input_filename
            xl = pd.ExcelFile(filename)
            missing = [s for s in ("User", "Manager") if s not in xl.sheet_names]
            if missing:
                raise ValueError(f"Missing required sheet(s): {', '.join(missing)}. "
                                "The workbook must contain both a 'User' and 'Manager' sheet.")
            user_df = pd.read_excel(xl, sheet_name="User")
            manager_df = pd.read_excel(xl, sheet_name="Manager")
            return user_df, manager_df
        except ValueError:
            raise
        except FileNotFoundError:
            raise ValueError(f"File '{filename}' not found. ...")
        except Exception as e:
            raise ValueError(f"Unexpected error reading Excel file: {e}. ...")
    
    def condition_dict(self, dataframe, cond_range=None):
        conditions = {}
        if cond_range:
            values = self.parse_range(cond_range)
            if not values:
                return conditions
        else:
            values = ['0', '50']
        for i in range(int(values[0]) - 1, int(values[1])):
            key = dataframe.iloc[i, 0]
            if pd.notna(dataframe.iloc[i, 1]):
                value = dataframe.iloc[i, 1:11].tolist()
                placings = dataframe.iloc[i, 13:16].tolist()
                value.extend(placings)
                conditions[key] = value
        return conditions
    
    def parse_range(self, cond_range):
        if cond_range is None or (isinstance(cond_range, float) and np.isnan(cond_range)) or str(cond_range).strip() == "":
            return [0,0]  # treat blank as no conditions
        s = str(cond_range).strip()
        # Try number-number
        matches = re.findall(r'(\d+)-(\d+)', s)
        if matches:
            start, end = map(int, matches[0])
            return [start, end]
        raise ValueError(f"Invalid condition range: {cond_range!r}")
    
    def separate_plates(self, user_df, manager_df):
        user_name = user_df.columns[1]
        nbcode = user_df.iloc[0,1]
        if manager_df.iloc[0,17] == '1 column':
            lc_column_number = 1
        elif manager_df.iloc[0,17] == '2 column':
            lc_column_number = 2
        else:
            raise ValueError(
                f"System type must either be \"1 column\" or \"2 column\", but got {manager_df.iloc[0,18]}"
            )
        plates = {}
        for i in [3, 21, 39]: # plates 1-3
            row_min = i
            name_row = i + 1

            name_values = user_df.iloc[name_row, [0, 1]]
            name = "_".join(str(x).strip() for x in name_values if pd.notna(x))
            plate = user_df.iloc[row_min:row_min+18, 3:28]

            # Validate plate values
            max_row = min(row_min + 18, len(user_df))
            max_col = min(28, user_df.shape[1])
            for r in range(row_min + 1, max_row):
                for c in range(4, max_col):
                    val = user_df.iloc[r, c]
                    if pd.notna(val):
                        if not isinstance(val, (int, float)) or not float(val).is_integer() or not (1 <= int(val) <= 50):
                            raise ValueError(
                                f"Invalid well value '{val}' in plate '{name}' at row {r+1}, column {c+1}: "
                                "must be an integer between 1 and 50."
                            )


            plate.columns = plate.iloc[0, :] # Set column names as index
            plate.index = plate.iloc[:, 0] # Set row names as index
            plate = plate.iloc[1:, 1:]
            if plate.isna().all().all():
                continue
            plates[name] = plate
        
        self.check_plate_colors(plates)

        wet_amounts = {}
        for i in range(0, 50):
            if pd.notna(manager_df.iloc[i, 1]):
                cond_num = manager_df.iloc[i, 0]
                well_amount = manager_df.iloc[i, 11]
                try:
                    if well_amount is None or np.isnan(well_amount) or well_amount == '':
                        wet_amounts[int(cond_num)] = 1
                    else:
                        wet_amounts[int(cond_num)] = int(well_amount)
                    if manager_df.iloc[i, 1] == "TrueBlank":
                        wet_amounts[int(cond_num)] = 10000  # TrueBlanks should be infinite
                except ValueError:
                    raise ValueError(f"Wet sample amount must be a whole number or blank. Check column 'Samples/well' for invalid entries.")
        num_to_run = {}
        for i in range(0, 50):
            if pd.notna(manager_df.iloc[i, 1]):
                cond_num = manager_df.iloc[i, 0]
                amt_to_run = manager_df.iloc[i, 12]
                if amt_to_run == 'all' or np.isnan(amt_to_run) or amt_to_run == '':
                    num_to_run[int(cond_num)] = 'all'
                else:
                    try:
                        num_to_run[int(cond_num)] = int(amt_to_run)
                    except ValueError:
                        raise ValueError("Number of samples to run must be either 'all' (no quotes), empty, or a whole number.")
        return nbcode, lc_column_number, wet_amounts, plates, num_to_run
    
    def check_plate_colors(self, plates):
        colors = []
        for name in plates:
            color = name.split("_")[0]
            colors.append(color)
        if not len(colors) == len(set(colors)):
            raise ValueError("Plate colors must be different.")
    
    def additional_info(self, user_df, manager_df):
        even = user_df.iloc[2,35] # "Yes" or "No", indicates if blocks should be forced to be even, sacrificing runs to do so
        TB_location = user_df.iloc[3,35] # indicates condition number for TrueBlanks
        experiment1 = user_df.iloc[6,35] # if not "All", which conditions belong to experiment 1
        experiment2 = user_df.iloc[7,35] # which conditions belong to experiment 2 if any
        lib_same = user_df.iloc[8,35] # "Yes" or "No", indicates if lib runs are the same for both experiments
        lib_placement = manager_df.iloc[3,17] # are lib runs before or after samples?
        SysValid_interval = manager_df.iloc[6,17] # how often to run system validation
        QC_frequency = manager_df.iloc[7,17] # how often should QC blocks be added
        lc_system = manager_df.iloc[10,17] # determines how the well
        return lib_placement, SysValid_interval, TB_location, experiment1, experiment2, lib_same, even, QC_frequency, lc_system
    
    def parse(self):
        try:
            user_df, manager_df = self.read_excel_to_dfs()
            nbcode, lc_number, wet_amounts, plates, num_to_run = self.separate_plates(user_df, manager_df)
            lib_placement, sysvalid_interval, TB_location, cond_range1, cond_range2, lib_same, even, qc_frequency, lc_system = self.additional_info(user_df, manager_df)
                                                    # cond_range1 = "All" or "{#}-{#}", cond_range2 = "" or "{#}-{#}", even = "Yes" or "No"
            TB_location = 'R5' if (TB_location == '' or TB_location != TB_location) else TB_location # check for blank or NaN and set to default if so
            even = 'Yes' if even not in ['Yes', 'No'] else even
            conditions = []
            conditions.append(self.condition_dict(manager_df)) # always include full condition dict
            if cond_range1.upper() != "ALL": # only include partial condition dicts if specified
                conditions.append(self.condition_dict(manager_df, cond_range1))
                conditions.append(self.condition_dict(manager_df, cond_range2))
            if lc_system == "Vanquish Neo":
                lc_symbol = ':'
            else: lc_symbol = ''
        except Exception as e:
            raise ValueError("Experiment conditions cannot be run due to missing or invalid configuration." \
            "Verify that all required experiment fields are correctly filled in the Excel sheet." \
            "If there is only one experiment the library values should be the same.")

        sample_type = "Unknown"  # or "Sample", etc.
        blank_method = "Blank_Method"  # fill in with real method if needed
        inj_vol = 1  # injection volume in µL

        self.blocker_info = [user_df, manager_df, lc_number, conditions, wet_amounts, plates, num_to_run,
                             lib_placement, sysvalid_interval, TB_location, cond_range1, cond_range2, lib_same, even, qc_frequency]
        self.output_info = [nbcode, lc_number, blank_method, sample_type, inj_vol, lc_symbol]

        return self.blocker_info, self.output_info