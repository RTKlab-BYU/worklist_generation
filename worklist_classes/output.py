import os
import pandas as pd

class Output:
    def __init__(self, parser_output, blocker_output):
        self.nbcode = parser_output[0]
        self.lc_number = parser_output[1]
        self.blank_method = parser_output[2]
        self.sample_type = parser_output[3]
        self.inj_vol = parser_output[4]
        self.conditions = parser_output[5][0]
        self.well_conditions = blocker_output[0]
        self.block_runs = blocker_output[1]
        self.positions = blocker_output[2]
        self.reps = blocker_output[3]
        self.msmethods = blocker_output[4]

    def create_filenames(self):
        # creates a list of filenames
        filenames = []
        TB_method_found = False # in a 2 column system we need to store the msmethod for a TrueBlank for later
        if self.lc_number == 1:
            columns = ['nbcode', 'conditions', 'block and run', 'position', 'rep','msmethod']
        elif self.lc_number == 2:
            columns = ['nbcode', 'conditions', 'block and run', 'position', 'rep', 'channel', 'msmethod']
        df = pd.DataFrame(columns=columns)
        for index, block_run in enumerate(self.block_runs):
            try:
                condition = self.conditions[self.well_conditions[index]][1]
            except KeyError:
                raise KeyError(f"Condition {self.well_conditions[index]} not found in conditions dictionary.")
            if self.lc_number == 1:
                df.loc[len(df)] = [self.nbcode, condition, self.block_runs[index],
                        self.positions[index], f"rep{self.reps[index]}", self.msmethods[index]]
            elif self.lc_number == 2:
                df.loc[len(df)] = [self.nbcode, condition, self.block_runs[index],
                        self.positions[index], f"rep{self.reps[index]}", f"ch{(index%2)+1}", self.msmethods[index]]
                if condition == "TrueBlank":
                    TB_method = self.msmethods[index]
                    TB_method_found = True
        for _, row in df.iterrows():
            joined = "_".join(str(x).strip() for x in row if pd.notna(x))
            filenames.append(joined)
        if TB_method_found == True:
            return filenames, TB_method
        return filenames, None

    def create_instrument_methods(self, methodpaths, methods, csv_file):
        inst_methods = []
        counter = 0
        for index, path in enumerate(methodpaths):
            if csv_file == 'LC':
                if self.lc_number == 2:
                    if counter%2 == 0:
                        methods[index] = "ChannelA_" + methods[index]
                    elif counter%2 == 1:
                        methods[index] = "ChannelB_" + methods[index]
            inst_methods.append("\\".join([path, methods[index]]))
            counter+=1
        return inst_methods

    def final_csv_format_as_pd(self, csv_file, filenames, well_conditions, positions, TB_method):
        # Create instrument methods for MS and LC
        method_paths = []
        method_names = []
        data_paths = []
        for index in well_conditions:
            if index not in self.conditions:
                raise KeyError(f"Condition {index} not found.")
            if len(self.conditions[index]) < 10:
                raise ValueError(f"Condition {index} is malformed: expected at least 10 fields, but got {len(self.conditions[index])}. Check the corresponding row in your Excel sheet for missing values.")
            if csv_file == 'MS':
                data_paths.append(self.conditions[index][2])
                method_paths.append(self.conditions[index][3])
                method_names.append(self.conditions[index][5])
            elif csv_file == 'LC':
                data_paths.append(self.conditions[index][6])
                method_paths.append(self.conditions[index][7])
                method_names.append(self.conditions[index][9])
        inst_methods = self.create_instrument_methods(method_paths, method_names, csv_file)
        # Offset for 2 column system
        if self.lc_number == 2:
            filenames.insert(0, f"{self.nbcode}_Preblank2")
            filenames.insert(0, f"{self.nbcode}_Preblank1")
            data_paths.insert(0, data_paths[0])
            data_paths.insert(0, data_paths[0])
            if csv_file == 'MS':
                # inst_methods.insert(0, blank_method)
                # inst_methods.insert(0, blank_method)
                inst_methods.insert(0, TB_method)
                inst_methods.insert(0, TB_method)
            elif csv_file == 'LC':
                inst_methods.append(inst_methods[-1])
                inst_methods.append(inst_methods[-1])
            positions.append(positions[-1])
            positions.append(positions[-1])

        if not (len(filenames) == len(data_paths) == len(inst_methods) == len(positions)):
            raise IndexError("Mismatched lengths when creating CSV data.")
        
        df = pd.DataFrame({
            "Sample Type": [self.sample_type] * len(filenames),
            "File Name": filenames,
            "Path": data_paths,
            "Instrument Method": inst_methods,
            "Position": positions,
            "Inj Vol": [self.inj_vol] * len(filenames)
        })

        # Convert to list-of-lists with exactly 6 columns
        data_rows = df.values.tolist()

        rows = []
        rows.append(["Bracket Type=4", "", "", "", "", ""])
        rows.append(["Sample Type", "File Name", "Path", "Instrument Method", "Position", "Inj Vol"])
        rows.extend(data_rows)

        # Build DataFrame with no column labels (just positional)
        return pd.DataFrame(rows)
    
    def putout(self):
        filenames, TB_method = self.create_filenames()
        # Create and export MS CSV
        ms_pd = self.final_csv_format_as_pd("MS", filenames.copy(), self.well_conditions.copy(), self.positions.copy(), TB_method)
        # Create and export LC CSV
        lc_pd = self.final_csv_format_as_pd("LC", filenames.copy(), self.well_conditions.copy(), self.positions.copy(), TB_method)
        #The files stored in files/output contain what needs to be sent to the mass spec and lc

        ms_filename = f"{self.nbcode}_MS.csv"
        lc_filename = f"{self.nbcode}_LC.csv"

        return ms_pd, lc_pd, ms_filename, lc_filename