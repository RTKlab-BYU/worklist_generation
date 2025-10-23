import random
import pandas as pd
import numpy as np
import sys
import math
import re

class Blocker:
    def __init__(self, parser_output):

        self.user_df = parser_output[0]
        self.manager_df = parser_output[1]
        self.lc_number = parser_output[2]
        self.all_conditions = parser_output[3][0]
        self.conditions1 = parser_output[3][1] if len(parser_output[3]) > 1 else None
        self.conditions2 = parser_output[3][2] if len(parser_output[3]) > 2 else None
        self.wet_amounts = parser_output[4]
        self.plates = parser_output[5]
        self.num_to_run = parser_output[6]
        self.lib_placement = parser_output[7]
        self.sysvalid_interval = parser_output[8]
        self.cond_range1 = parser_output[9]
        self.cond_range2 = parser_output[10]
        self.lib_same = parser_output[11]
        self.even = parser_output[12]
        self.qc_frequency = parser_output[13]
        self.final_output = []
        self.output_folder = None

    def generate_seed(self, run_seed=None):
        random.seed(run_seed := run_seed or random.randrange(sys.maxsize))
        print(run_seed)

    def safe_int(self, val, default=0):
        try:
            if pd.isna(val):
                return default
            return int(val)
        except (ValueError, TypeError):
            return default

    def check_for_trueblank(self, conditions1, conditions2=None):
        found = False
        for i, cond in enumerate(conditions1.items()):
            if cond[0] == "TrueBlank":
                found = True
                break
        if not found and conditions2:
            for i, cond in enumerate(conditions2.items()):
                if cond[0] == "TrueBlank":
                    found = True
                    break
        if found:
            if not conditions2:
                return conditions1, i+1, True # condition IDs are 1-indexed
            else:
                return conditions1, conditions2, i+1, True # condition IDs are 1-indexed
        conditions1[len(conditions1)] = ["TrueBlank", "TrueBlank", "", "", "", "", "", "", "", "", 0, 0, 0, 0, 0, 0]
        if not conditions2:
            return conditions1, len(conditions1), False # return the new condition ID
        else:
            return conditions1, conditions2, len(conditions1), False
        
    # Test this alternate shorter function to see if it works   
    # def check_for_trueblank(conditions1, conditions2=None):
    #     for i, (k, _) in enumerate(conditions1.items()):
    #         if k == "TrueBlank":
    #             return (conditions1, i+1, True) if not conditions2 else (conditions1, conditions2, i+1, True)
    #     if conditions2:
    #         for i, (k, _) in enumerate(conditions2.items()):
    #             if k == "TrueBlank":
    #                 return conditions1, conditions2, i+1, True
    #     conditions1[len(conditions1)] = ["TrueBlank", "TrueBlank", "", "", "", "", "", "", "", "", 0, 0, 0, 0, 0, 0]
    #     return (conditions1, len(conditions1), False) if not conditions2 else (conditions1, conditions2, len(conditions1), False)

        
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

    def process_plate(self, plate, plate_name, wet_amounts, cond_range = None):
        wells_list = []
        RGB = plate_name.split("_")[0]
        if cond_range:
            try:
                ranges = self.parse_range(cond_range)
            except TypeError:
                ranges = ['0', '0'] # range was defined as blank so conditions should return blank

        #     newplate = plate.iloc[int(ranges[0]):int(ranges[1])+1, :].copy()
        for r_idx, row_series in plate.iterrows():
            for c_idx, col_name in enumerate(plate.columns):
                try:
                    col_num = int(col_name)
                except ValueError:
                    raise ValueError(f"Column name '{col_name}' in plate '{plate_name}' must be a unique integer (e.g., 1, 2, 3). Please rename the column accordingly.")
                value = row_series[col_name]
                if pd.isna(row_series[col_name]):
                    continue
                plate_location = f"{r_idx}{col_num}" # well location (e.g. A1)
                abs_location = RGB + plate_location
                try:
                    try:
                        # Try to coerce to float first, whether it's a number or a numeric string
                        as_float = float(value)
                        if math.isfinite(as_float) and as_float.is_integer():
                            normalized_value = int(as_float)
                        else:
                            raise ValueError
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"Invalid condition ID '{value}' at location {abs_location}. "
                            "Condition IDs must be whole numbers without extra symbols."
                        )
                    for i in range(0, int(wet_amounts[normalized_value])):
                        if cond_range:
                            if normalized_value in range(int(ranges[0]), int(ranges[1]) + 1):
                                wells_list.append([normalized_value, abs_location]) # value, absolute location (e.g. RA1)
                        else:
                            wells_list.append([normalized_value, abs_location]) # value, absolute location (e.g. RA1)
                except KeyError:
                    if cond_range:
                        continue # only adds wells to wells_list associated with the conditions specified in cond_range
                    else:
                        raise KeyError(f'ERROR: You may have a sample in your plate that is not labeled in the conditions legend.')
        return wells_list

    def compare_wells_and_counts(self, wells_list, conditions, wet_amounts):
        if wet_amounts is None:
            raise ValueError("Samples/well must be provided for validation")

        QC_num, wet_QC_num, Blank_num, TrueBlank_num, Lib_num, SysValid_num = [], [], [], [], [], []

        # Assign condition IDs to the right bucket
        for key, row in conditions.items():
            label = row[0]
            try:
                key_int = int(key)
            except (ValueError, TypeError):
                continue  # skip malformed keys

            if label == 'QC':
                QC_num.append(key_int)
            elif label == 'WetQC':
                wet_QC_num.append(key_int)
            elif label == 'Blank':
                Blank_num.append(key_int)
            elif label == 'TrueBlank':
                TrueBlank_num.append(key_int)
            elif label == 'Lib':
                Lib_num.append(key_int)
            elif label == 'SystemValidation':
                SysValid_num.append(key_int)

        # Count how many wells we actually have for each condition type
        total_QC = sum(1 for w in wells_list if w[0] in QC_num)
        total_wet_QC = sum(1 for w in wells_list if w[0] in wet_QC_num)
        total_Blank = sum(1 for w in wells_list if w[0] in Blank_num)
        total_TrueBlank = sum(1 for w in wells_list if w[0] in TrueBlank_num)
        total_Lib = sum(1 for w in wells_list if w[0] in Lib_num)
        total_SysValid = sum(1 for w in wells_list if w[0] in SysValid_num)

        # Helper to safely extract spacing values (before/after/between)
        def spacing_sum(cond_row):
            vals = []
            for idx in (10, 11, 12):
                try:
                    vals.append(int(cond_row[idx]) if cond_row[idx] not in (None, "", "NaN") else 0)
                except (IndexError, ValueError, TypeError):
                    vals.append(0)
            return sum(vals)

        # Validator: check actual wells * wet_amounts vs expected spacings
        def validate(cond_ids, total_count, label, infinite_source=False):
            for num in cond_ids:
                expected = spacing_sum(conditions[num])
                actual = total_count * wet_amounts.get(num, 1)

                if infinite_source:
                    # Only need at least 1, no matter how many times it's referenced
                    if total_count < 1:
                        raise ValueError(f"Error: No {label} wells found, need at least 1.")
                else:
                    if actual < expected:
                        raise ValueError(
                            f"Error: Not enough {label} wells. "
                            f"Expected at least {expected}, but found {actual}."
                        )


        # Run checks for each type
        validate(QC_num, total_QC, "QC")
        validate(wet_QC_num, total_wet_QC, "WetQC")
        validate(Blank_num, total_Blank, "Blank")
        validate(TrueBlank_num, total_TrueBlank, "TrueBlank", infinite_source=True)
        validate(Lib_num, total_Lib, "Library", infinite_source=True)
        validate(SysValid_num, total_SysValid, "SystemValidation", infinite_source=True)


    def column_sorter(self, wells_list, conditions, wet_amounts, num_to_run, lc_number, Lib_placement, lib_same, cond_range1): #split the wells list evenly between the two columns
        column1 = []
        column2 = []
        extras = [] # these are the odds ones out to attach at the end to run anyways if wanted
        nonsample_before = []
        nonsample_after = []
        nonsample_other = []
        QC_num, wet_QC_num, Blank_num, TrueBlank_num, Lib_num, SysValid_num = [], [], [], [], [], []

        # find number associated with each nonsample well type
        list_of_keys = list(conditions.keys())
        for key in list_of_keys:
            if conditions[key][0] == 'QC':
                QC_num.append(int(key))
            if conditions[key][0] == 'WetQC':
                wet_QC_num.append(int(key))
            if conditions[key][0] == 'Blank':
                Blank_num.append(int(key))
            if conditions[key][0] == 'TrueBlank':
                TrueBlank_num.append(int(key))
            if conditions[key][0] == 'Lib':
                Lib_num.append(int(key))
            if conditions[key][0] == "SystemValidation":
                SysValid_num.append(int(key))

        # remove non sample wells into new list
        QC_list, wet_QC_list, Blank_list, TrueBlank_list, Lib_list, SysValid_list = [], [], [], [], [], []
        
        for well in wells_list[:]: # don't change this! iterates over copy of original list and edits original
            if QC_num:
                for num in QC_num:
                    if int(well[0]) == num:
                        QC_list.append(well)
                        wells_list.remove(well)
            if wet_QC_num:
                for num in wet_QC_num:
                    if well[0] == num:
                        wet_QC_list.append(well)
                        wells_list.remove(well)
            if Blank_num:
                for num in Blank_num:
                    if well[0] == num:
                        Blank_list.append(well)
                        wells_list.remove(well)
            if TrueBlank_num:
                for num in TrueBlank_num:
                    if well[0] == num:
                        TrueBlank_list.append(well)
                        wells_list.remove(well)
                        continue
            if Lib_num:
                for num in Lib_num:
                    if well[0] == num:
                        Lib_list.append(well)
                        wells_list.remove(well)
            if SysValid_num:
                for num in SysValid_num:
                    if well[0] == num:
                        SysValid_list.append(well)
                        wells_list.remove(well)

        # randomize the nonsamples
        random.shuffle(QC_list)
        random.shuffle(wet_QC_list)
        random.shuffle(Blank_list)
        random.shuffle(TrueBlank_list)
        random.shuffle(Lib_list)
        random.shuffle(SysValid_list)

        # move a well to 'extras' if a list has an odd number of wells and it uses a 2 column system
        if lc_number == 2:
            if len(QC_list) %2 != 0:
                extras.append(QC_list[-1:])
                QC_list = QC_list[:-1]
            if len(wet_QC_list) %2 != 0:
                extras.append(wet_QC_list[-1:])
                wet_QC_list = wet_QC_list[:-1]
            if len(Blank_list) %2 != 0:
                extras.append(Blank_list[-1:])
                Blank_list = Blank_list[:-1]
            if len(TrueBlank_list) %2 != 0:
                extras.append(TrueBlank_list[-1:])
                TrueBlank_list = TrueBlank_list[:-1]
            if len(Lib_list) %2 != 0:
                extras.append(Lib_list[-1:])
                Lib_list = Lib_list[:-1]

        # divide them into their respective nonsample lists based on user inputs
        # add nonsamples to 'nonsample_before' list
        if QC_list:
            for cond in QC_num: # QC_num is a list of the condition numbers of all QCs
                one_QC = [] # temporary list to sort different QCs by their number
                for well in QC_list:
                    if well[0] == cond:
                        one_QC.append(well)
                nonsample_before.append(one_QC[:self.safe_int(conditions[cond][10], default=0)])
                for well in one_QC[:self.safe_int(conditions[cond][10], default=0)]:
                    QC_list.remove(well)
            #QC_list = QC_list[int(conditions[cond][10]):]
        if wet_QC_list:
            for cond in wet_QC_num:
                one_QC = []
                for well in wet_QC_list:
                    if well[0] == cond:
                        one_QC.append(well)
                nonsample_before.append(wet_QC_list[:self.safe_int(conditions[cond][10], default=0)])
                for well in one_QC[:self.safe_int(conditions[cond][10], default=0)]:
                    wet_QC_list.remove(well)
            #wet_QC_list = wet_QC_list[int(conditions[cond][10]):]
            #nonsample_before.append(wet_QC_list[:spacings[1][0]])
            #wet_QC_list = wet_QC_list[spacings[1][0]:]
        if Blank_list:
            for cond in Blank_num:
                one_QC = []
                for well in Blank_list:
                    if well[0] == cond:
                        one_QC.append(well)
                nonsample_before.append(Blank_list[:self.safe_int(conditions[cond][10], default=0)])
                for well in one_QC[:self.safe_int(conditions[cond][10], default=0)]:
                    Blank_list.remove(well)
            #Blank_list = Blank_list[int(conditions[cond][10]):]
            # nonsample_before.append(Blank_list[:spacings[2][0]])
            # Blank_list = Blank_list[spacings[2][0]:]
        if Lib_list and Lib_placement == "Before" and cond_range1.upper() == "ALL": #controls library placement
            nonsample_before.append(Lib_list)
        if TrueBlank_list:
            for cond in TrueBlank_num: # TrueBlank_num is a list of the condition numbers of all TrueBlanks
                one_QC = []
                for well in TrueBlank_list:
                    if well[0] == cond:
                        one_QC.append(well)
                for i in range(0, self.safe_int(conditions[cond][10], default=0)):
                    nonsample_before.append(TrueBlank_list[:1])
        # add nonsamples to 'nonsample_after' list
        if QC_list:
            for num in QC_num:
                num_list = [well for well in QC_list if well[0] == num]
                nonsample_after.append(num_list[:self.safe_int(conditions[num][11], default=0)])
                for well in num_list[:self.safe_int(conditions[num][11], default=0)]:
                    QC_list.remove(well)
        if wet_QC_list:
            for num in wet_QC_num:
                num_list = [well for well in wet_QC_list if well[0] == num]
                nonsample_after.append(num_list[:self.safe_int(conditions[num][11], default=0)])
                for well in num_list[:self.safe_int(conditions[num][11], default=0)]:
                    wet_QC_list.remove(well)
        if Blank_list:
            for num in Blank_num:
                num_list = [well for well in Blank_list if well[0] == num]
                nonsample_after.append(num_list[:self.safe_int(conditions[num][11], default=0)])
                for well in num_list[:self.safe_int(conditions[num][11], default=0)]:
                    Blank_list.remove(well)
        if Lib_list and Lib_placement == "After" and cond_range1 == "ALL": #controls library placement
            nonsample_after.append(Lib_list)
        if TrueBlank_list:
            for num in TrueBlank_num:
                num_list = [well for well in TrueBlank_list if well[0] == num]
                nonsample_after.append(num_list[:self.safe_int(conditions[num][11], default=0)])
                for well in num_list[:self.safe_int(conditions[num][11], default=0)]:
                    TrueBlank_list.remove(well)

        # code for nonsample_other was removed

        if QC_list:
            for well in QC_list:
                nonsample_other.append(well)
        if wet_QC_list:
            for well in wet_QC_list:
                nonsample_other.append(well)
        if Blank_list:
            for well in Blank_list:
                nonsample_other.append(well)
        if TrueBlank_list:
            for well in TrueBlank_list:
                nonsample_other.append(well)


        # if library runs are not the same in a two experiment plate, they must be returned separately
        separate_Lib = []
        if Lib_list and cond_range1.upper() != "ALL":
            separate_Lib.append(Lib_list)

        ### handle divide sample lists into columns ###

        # remove the values from conditions dict that are in the between blocks + remove System Validation wells
        sample_keys = list(conditions.keys())
        new_keys = sample_keys.copy()
        between_keys = [QC_num, Blank_num, TrueBlank_num, Lib_num, wet_QC_num, SysValid_num]
        for sample in sample_keys:
        # unwrap NumPy scalar to plain int if needed
            sample_val = int(sample) if hasattr(sample, "item") else sample  

            # flatten between_keys so it's just numbers
            flat_between = [x for sub in between_keys for x in sub]

            if sample_val in flat_between:
                new_keys.remove(sample)

        sample_dict = {}
        for sample in new_keys:
            #Count number of sample wells in wells list
            count = 0
            for well in wells_list:
                if well[0] == sample:
                    count += 1
            sample_dict[sample] = [conditions[sample], count] # key is number from well plate, value is list containg
        # info from conditions then number of wells of that sample

        # put samples into respective lists and randomize lists
        for sample in sample_dict.keys(): # sample is an integer
            sample_list = []
            for well in wells_list:
                if sample == well[0]:
                    sample_list.append(well)
            # randomize list of wells
            random.shuffle(sample_list)
            # reduce number of samples to that specified in num_to_run dictionary
            if type(num_to_run[sample]) == int:
                sample_list = sample_list[:num_to_run[sample]]

            if lc_number == 2:
                # remove extra samples
                if len(sample_list) %2 != 0:
                    extras.append(sample_list[-1:])
                    sample_list = sample_list[:-1]
                # divide the samples evenly between the two columns
                even = True
                for sample in sample_list:
                    if even == True:
                        column1.append(sample)
                        even = False
                    elif even == False:
                        column2.append(sample)
                        even = True
            elif lc_number == 1:
                for sample in sample_list:
                    column1.append(sample)
        
        if lc_number ==2:
            return (nonsample_before, nonsample_after, nonsample_other, column1, column2, extras, SysValid_list, separate_Lib)

        elif lc_number == 1:
            #column1 = wells_list
            return (nonsample_before, nonsample_after, nonsample_other, column1, SysValid_list, separate_Lib)

    def blocker(self, conditions, even, column1, column2 = None):
        if column2:
            columns = [column1, column2]
        if not column2:
            columns = [column1]
        both_blocks = []
        extras_dict = {} #stores uneven samples so that the second column can use same block assignments
        for column in columns:

            #creates a dictionary that stores how much of each sample is in the column so it can be blocked
            sample_dict = {}
            for well in column:
                if well[0] not in sample_dict.keys():
                    sample_dict[well[0]] = 1
                elif well[0] in sample_dict.keys():
                    sample_dict[well[0]] += 1
            sample_amounts = list(sample_dict.values())
            try:
                sample_block_num = min(sample_amounts)
            except ValueError:
                print("No conditions were added to the plate.")
                return([[[]]], 0)

            sample_keys = list(sample_dict.keys())
            for sample in sample_keys:
                #Count number of sample wells in wells list
                count = 0
                for well in column:
                    if well[0] == sample:
                        count += 1
                to_add = count // sample_block_num
                sample_dict[sample] = [conditions[sample], to_add, count]
            # create list of sample well blocks
            sample_blocks = []
            blocks_created = 0
            while blocks_created < sample_block_num:
                block = [] # create each block for the samples
                for sample in sample_dict.keys(): # go through each sample, create temporary list
                    sample_list = []
                    for well in column:
                        if well[0] == sample:
                            sample_list.append(well)
                    # append these separately
                    #block.append(sample_list[:sample_dict[sample][1]])
                    wells_to_append = sample_list[:sample_dict[sample][1]]
                    for well in wells_to_append:
                        block.append(well)
                for item in block:
                    if item in column:
                        column.remove(item)
                random.shuffle(block)
                sample_blocks.append(block)
                blocks_created += 1
            num_of_blocks = len(sample_blocks)
            if blocks_created == sample_block_num and even.upper() == "NO":
                # assigns leftover samples randomly to blocks if the user does not want even blocks
                num_of_blocks = len(sample_blocks)
                for sample in sample_dict.keys(): # go through each sample, create temporary list
                    sample_list = []
                    for well in column:
                        if well[0] == sample:
                            sample_list.append(well)
                    for item in sample_list:
                        if column == column1:
                            placement = random.randint(0, num_of_blocks-1)
                            sample_blocks[placement].append(item)
                            try:
                                extras_dict[item[0]][0] += 1
                                extras_dict[item[0]][1].append(placement)
                            except KeyError:
                                extras_dict[item[0]] = [1,[]]
                                extras_dict[item[0]][1].append(placement)
                        else:
                            placement = extras_dict[item[0]][1][-1]
                            extras_dict[item[0]][1] = extras_dict[item[0]][1][:-1]
                            sample_blocks[placement].append(item)
                            extras_dict[item[0]][0] -= 1
            for block in sample_blocks:
                random.shuffle(block)
            both_blocks.append(sample_blocks)
        return(both_blocks, num_of_blocks)

    def nonsample_blocker(self, lc_number, nonsample_other, num_of_blocks, conditions, even):
        """ Will divide the QC, Blanks, Trueblanks etc, reserved to be between the runs, into blocks
            Make sure that number of nonsample blocks does not exceed number of sample blocks
            These blocks should not be randomized"""
        nonsample_other = [item for block in nonsample_other for item in block] # flattens list

        sample_dict = {}
        for well in nonsample_other:
            if well[0] not in sample_dict.keys():
                sample_dict[well[0]] = 1
            elif well[0] in sample_dict.keys():
                sample_dict[well[0]] += 1
        sample_amounts = list(sample_dict.values())
        if sample_amounts:
            if lc_number == 2:
                nonsample_block_num = min(sample_amounts) // 2
            elif lc_number == 1:
                nonsample_block_num = min(sample_amounts)
        else:
            nonsample_block_num = 0
        # '//2' ensures that nonsamples can come in pairs of two by
        # preventing blocks of only one of each nonsample condition

        # # set 'sample_block_num' to correct number, max is one less than the sample_block number from 'blocker' function
        # if num_of_blocks == 1:
        #     block_num = 1
        # elif num_of_blocks == 0:
        #     block_num = nonsample_block_num
        # elif nonsample_block_num == 0:
        #     block_num = num_of_blocks
        # elif nonsample_block_num < num_of_blocks: #and even.upper() == "YES": ## this might need to be changed
        #     block_num = nonsample_block_num
        # elif nonsample_block_num >= num_of_blocks:
        #     block_num = num_of_blocks - 1
        block_num = nonsample_block_num # temporary! write a new calculation using total number of 
        # sample blocks and QC frequency

        sample_keys = list(sample_dict.keys())
        for sample in sample_keys:
            #Count number of sample wells in wells list
            count = 0
            for well in nonsample_other:
                if well[0] == sample:
                    count += 1
            to_add = count // block_num
            if to_add == 1 and lc_number == 2:
                raise ValueError("If you want to include nonsample wells between sample/condition wells in a two-column layout, you must include at least two of each nonsample well type.")
            sample_dict[sample] = [conditions[sample], to_add, count]
        nonsample_blocks = []
        blocks_created = 0
        while blocks_created < block_num:
            block = [] # create each block for the samples
            for sample in sample_dict.keys(): # go through each sample, create temporary list
                sample_list = []
                for well in nonsample_other:
                    if well[0] == sample:
                        sample_list.append(well)
                #block.append(sample_list[:sample_dict[sample][1]]) # problem?
                wells_to_append = sample_list[:sample_dict[sample][1]]
                for well in wells_to_append:
                    block.append(well)
            for item in block:
                if item in nonsample_other:
                    nonsample_other.remove(item)
            #random.shuffle(block)
            nonsample_blocks.append(block)
            blocks_created += 1
        if blocks_created == block_num and even.upper() == "NO":
            # assigns leftover samples randomly to blocks
            num_of_blocks = len(nonsample_blocks)
            for sample in sample_dict.keys(): # go through each sample, create temporary list
                sample_list = []
                for well in nonsample_other:
                    if well[0] == sample:
                        sample_list.append(well)
                for item in sample_list:
                    placement = random.randint(0, num_of_blocks-1)
                    nonsample_blocks[placement].append(item)
        print(f'Number of nonsample blocks: {len(nonsample_blocks)}')
        return(nonsample_blocks)

    def zipper(self, both_blocks): # zips column1 and column2 together
        if len(both_blocks) == 1:
            both_blocks = [item for block in both_blocks for item in block]
            return both_blocks
        else:
            column1 = both_blocks[0]
            column2 = both_blocks[1]
            sample_blocks = []
            max_length = max(len(column1), len(column2))
            for i in range(0, max_length):
                block = []
                for x in range(0, len(column1[i])):
                    block.append(column1[i][x])
                    block.append(column2[i][x])
                sample_blocks.append(block)
        return sample_blocks

    def non_sample_lists(self, conditions, wells_list, blocks_to_make): # all inputs exist 10/13/2025
        QC_num, wet_QC_num, Blank_num, TrueBlank_num = [], [], [], []
        # find number associated with each nonsample well type
        list_of_keys = list(conditions.keys())
        for key in list_of_keys:
            if conditions[key][0] == 'QC':
                QC_num.append(int(key))
            if conditions[key][0] == 'WetQC':
                wet_QC_num.append(int(key))
            if conditions[key][0] == 'Blank':
                Blank_num.append(int(key))
            if conditions[key][0] == 'TrueBlank':
                TrueBlank_num.append(int(key))
        # remove non sample wells into new list
        QC_list = []
        wet_QC_list = []
        Blank_list = []
        TrueBlank_list = []
        for well in wells_list[:]: # don't change this! iterates over copy of original list and edits original
            if QC_num:
                for num in QC_num:
                    if int(well[0]) == num: # this is the current error
                        QC_list.append(well)
                        wells_list.remove(well)
            if wet_QC_num:
                for num in wet_QC_num:
                    if well[0] == num:
                        wet_QC_list.append(well)
                        wells_list.remove(well)
            if Blank_num:
                for num in Blank_num:
                    if well[0] == num:
                        Blank_list.append(well)
                        wells_list.remove(well)
            if TrueBlank_num:
                for num in TrueBlank_num:
                    if well[0] == num:
                        TrueBlank_list.append(well)
                        wells_list.remove(well)
        list_nonsample_blocks = []
        nonsample_objects = [[QC_list, QC_num], [wet_QC_list, wet_QC_num], [Blank_list, Blank_num], [TrueBlank_list, TrueBlank_num]]
        #raise ValueError(f'Check nonsample_obects: {nonsample_objects}')
        # check number of items to add
        adding_check = []
        for i in range(0, blocks_to_make):
            nonsample_block = []
            for pair in nonsample_objects:
                if pair[0]:
                    for num in pair[1]:
                        num_list = [well for well in pair[0] if well[0] == num]
                        nonsample_block.extend(num_list[0:self.safe_int(conditions[num][12], default=0)])
                        adding_check.append(self.safe_int(conditions[num][12], default=0))
                        for well in num_list[:self.safe_int(conditions[num][12], default=0)]:
                            pair[0].remove(well)
            list_nonsample_blocks.append(nonsample_block)
        return list_nonsample_blocks

    def combine_samples_and_nonsamples(self, nonsample_before, nonsample_after, sample_blocks, non_sample_other, QC_frequency, conditions):
        # flatten sample blocks
        samples_flat = [well for block in sample_blocks for well in block]
        ## create nonsample blocks
        # count # of samples divide by frequency to # of blocks
        num_samples = len(samples_flat)
        blocks_to_make = num_samples // QC_frequency

        # create blocks and put into a list
        list_nonsample_blocks = self.non_sample_lists(conditions, non_sample_other, blocks_to_make)

        # assume that at this point non_sample_blocks are reformatted correctly
        samples_and_non = []
        nonsample_before = [well for block in nonsample_before for well in block]
        if nonsample_before:
            samples_and_non.append(nonsample_before)

        temp_samples = []
        counter = 0
        for well in samples_flat[:]:
            temp_samples.append(well)
            counter += 1
            if counter == QC_frequency:
                samples_and_non.append(temp_samples)
                temp_samples = []
                try:
                    samples_and_non.append(list_nonsample_blocks[0])
                except IndexError:
                    raise IndexError("Not enough QC groups were added to the plate.")
                list_nonsample_blocks = list_nonsample_blocks[1:]
                counter = 0
        if temp_samples:
            samples_and_non.append(temp_samples)

        nonsample_after = [well for block in nonsample_after for well in block]
        if nonsample_after:
            samples_and_non.append(nonsample_after)

        return samples_and_non
            
    def blocknum_and_flatten(self, non_flat_list):
        # non_flat_list is a list of lists
        # add block numbers to each of the wells
        for index, block in enumerate(non_flat_list):
            index += 1
            #for part in block:
            for well in block:
                if isinstance(well, list) and len(well) < 3:
                    well.append(f"blo{index-1}") # fix this, see if it blocks after this to know if there really is only one block
        # flatten
        flat_list = [well for block in non_flat_list for well in block]
        return(flat_list)



    def block_zipper(self, nonsample_before, nonsample_after, sample_blocks, non_sample_blocks, even):
        final_flat_list = []
        # Add the pre-block if provided
        if nonsample_before:
            final_flat_list.append(nonsample_before)

        total_blocks = len(sample_blocks) + len(non_sample_blocks)
        sample_i = 0
        nonsample_i = 0

        if even.upper() == "NO":
            print("NOT EVEN")
            # Interleave using mostly even spacing
            for i in range(total_blocks):
                # Compute where the next non-sample block *should* go based on spacing
                expected_nonsample_pos = (nonsample_i + 1) * total_blocks / (len(non_sample_blocks) + 1) if non_sample_blocks else float('inf')

                if i + 1 >= expected_nonsample_pos and nonsample_i < len(non_sample_blocks):
                    final_flat_list.append([non_sample_blocks[nonsample_i]])
                    nonsample_i += 1
                elif sample_i < len(sample_blocks):
                    final_flat_list.append([sample_blocks[sample_i]])
                    sample_i += 1

        if even.upper() == "YES":
            print("EVEN")
            if not non_sample_blocks and not sample_blocks:
                final_flat_list.append([[]])
            elif not non_sample_blocks:
                final_flat_list.append(sample_blocks)
            elif not sample_blocks:
                final_flat_list.append(non_sample_blocks)
            # Interleave with perfectly even spacing
            else:
                sample_interval = len(sample_blocks) // len(non_sample_blocks) 
                if sample_interval >= 1:
                    for i in range(0, len(non_sample_blocks) - sample_interval):
                        for i in range(0, sample_interval):
                            final_flat_list.append([sample_blocks[sample_i]])
                            sample_i += 1
                        final_flat_list.append([non_sample_blocks[nonsample_i]])
                        nonsample_i += 1
                    for i in range(0, sample_interval):
                        final_flat_list.append([sample_blocks[sample_i]])
                        sample_i += 1
                else:
                    sample_interval = len(non_sample_blocks) // len(sample_blocks)
                    for i in range(0, len(sample_blocks) - sample_interval):
                        for i in range(0, sample_interval):
                            final_flat_list.append([sample_blocks[sample_i]])
                            sample_i += 1
                        final_flat_list.append([non_sample_blocks[nonsample_i]])
                        nonsample_i += 1
                    for i in range(0, sample_interval):
                        final_flat_list.append([sample_blocks[sample_i]])
                        sample_i += 1

        # Add the post-block if provided
        if nonsample_after:
            final_flat_list.append(nonsample_after)
        
        print(final_flat_list)

        return final_flat_list

    def two_xp_zipper(self, flat_list_1, flat_list_2, two_xp_TB, conditions, two_xp_TB_location):
        # find a TrueBlank well
        TB_well = [two_xp_TB, two_xp_TB_location[0][1], "end"]

        if two_xp_TB is None or two_xp_TB == 'None':
            raise ValueError("Error: When more than one experiment is run on one worklist, a TrueBlank well must be specified in cell AM37 of the excel sheet.") # fix this
        # removes empty blocks from the lists
        flat_list_1 = [block for block in flat_list_1 if block != [[]]]
        flat_list_2 = [block for block in flat_list_2 if block != [[]]]

        # add block numbers to each of the wells
        for index, block in enumerate(flat_list_1):
            index += 1
            for part in block:
                for well in part:
                    if isinstance(well, list) and len(well) < 3:
                        well.append(f"blo{index-1}") # fix this, see if it blocks after this to know if there really is only one block
        for index, block in enumerate(flat_list_2):
            index += 1
            for part in block:
                for well in part:
                    if isinstance(well, list) and len(well) < 3:
                        well.append(f"blo{index-1}")

        # flattens the lists
        flat_list_1 = self.flattener(flat_list_1)
        flat_list_2 = self.flattener(flat_list_2)

        # forces lists to be the same length by adding TrueBlank wells
        if len(flat_list_1) > len(flat_list_2):
            to_add = len(flat_list_1) - len(flat_list_2)
            for i in range(0, to_add):
                flat_list_2.append(TB_well)
        elif len(flat_list_2) > len(flat_list_1):
            to_add = len(flat_list_2) - len(flat_list_1)
            for i in range(0, to_add):
                flat_list_1.append(TB_well)
        combined_list = []
        for i in range(0, len(flat_list_1)):
            combined_list.append(flat_list_1[i])
            combined_list.append(flat_list_2[i])
        return combined_list

    def flattener(self, final_list):
        """Flatten a three-layer nested list into a list of lists, ignoring empty lists."""
        while isinstance(final_list, list) and len(final_list) == 1 and isinstance(final_list[0], list):
            final_list = final_list[0]

        out = []
        for block in final_list:
            if not block:
                continue
            for group in block:
                if not group:
                    continue
                for pair in group:
                    if pair:  # skip completely empty elements
                        out.append(pair)
        return out

    def rep_tracker(self, flattened, conditions):
        reps = []
        rep_counters = {}
        for well in flattened:
            if isinstance(well, list) and len(well) > 2:
                try:
                    condition = int(well[0])
                    if condition in conditions:
                        if condition not in rep_counters:
                            rep_counters[condition] = 1

                        elif condition in rep_counters:
                            rep_counters[condition] += 1
                        reps.append(rep_counters[condition])
                    else:
                        raise KeyError(f"Condition {condition} not found in conditions dictionary.")
                except ValueError:
                    condition = well[0]
                    if condition in conditions:
                        if condition not in rep_counters:
                            rep_counters[condition] = 1

                        elif condition in rep_counters:
                            rep_counters[condition] += 1
                        reps.append(rep_counters[condition])
                    #raise ValueError(f"Invalid condition value '{well[0]}' in well {well}.")
            else:
                reps.append(None)  # If well is not a valid list or doesn't have enough elements
        return reps

    def attach_Lib(self, two_xp_flat_list, separate_Lib1, separate_Lib2, two_xp_TB, two_xp_TB_location, Lib_placement, lib_same):
        """Attach library experiments for two-experiment runs.
        Inserts TB wells every other position during library runs.
        """
        # Flatten nested lists
        separate_Lib1 = [item for sublist in separate_Lib1 for item in sublist] if separate_Lib1 else []
        separate_Lib2 = [item for sublist in separate_Lib2 for item in sublist] if separate_Lib2 else []

        # Tag wells with their library identity
        if separate_Lib1:
            for well in separate_Lib1:
                well.append("Lib1")
        if separate_Lib2:
            for well in separate_Lib2:
                well.append("Lib2")

        TB_well = [two_xp_TB, two_xp_TB_location[0][1], "end"]
        to_add = []

        def interleave_with_tb(lib):
            """Insert TB wells every other element of a library list."""
            seq = []
            for i, well in enumerate(lib):
                seq.append(well)
                seq.append(TB_well)  # always follow with TB
            return seq

        if lib_same.upper() == "NO":
            # Run libraries separately, TB every other
            if separate_Lib1:
                to_add.extend(interleave_with_tb(separate_Lib1))
            if separate_Lib2:
                to_add.extend(interleave_with_tb(separate_Lib2))

        elif lib_same.upper() == "YES":
            if separate_Lib1 and separate_Lib2:
                # Balance lengths
                if len(separate_Lib1) > len(separate_Lib2):
                    separate_Lib2.extend([TB_well] * (len(separate_Lib1) - len(separate_Lib2)))
                elif len(separate_Lib2) > len(separate_Lib1):
                    separate_Lib1.extend([TB_well] * (len(separate_Lib2) - len(separate_Lib1)))
                # Interleave across libraries with TBs
                for i in range(len(separate_Lib1)):
                    to_add.append(separate_Lib1[i])
                    to_add.append(TB_well)
                    to_add.append(separate_Lib2[i])
                    to_add.append(TB_well)
            elif separate_Lib1:
                to_add.extend(interleave_with_tb(separate_Lib1))
            elif separate_Lib2:
                to_add.extend(interleave_with_tb(separate_Lib2))

        else:  # Fallback
            if separate_Lib1:
                to_add.extend(interleave_with_tb(separate_Lib1))
            if separate_Lib2:
                to_add.extend(interleave_with_tb(separate_Lib2))
            # safety flush
            to_add.extend([TB_well, TB_well])
        # Place libraries before or after main experiment

        if Lib_placement == "Before":
            return to_add + two_xp_flat_list
        else:  # default After
            return two_xp_flat_list + to_add

    def insert_sysQC(self, flattened_list, SysValid_list, SysValid_interval, lc_number, two_xp_TB, two_xp_TB_location, conditions):
        # Counter to return how many more System Validation wells should be added
        missing_SV = 0
        SysVal_copy = SysValid_list.copy()
        # TB well is also needed here for SysVal QC in the midst of library runs
        TB_well = [two_xp_TB, two_xp_TB_location[0][1], "TrueBlank"]
        # System QC wells will need a block indicator, for now they will be labeled as SysQC in place of 'blo{#}' format
        if len(SysValid_list) == 0:
            print("No System Validation QC were labeled on and/or inserted in the plate.")
            return(flattened_list)
        SysValid_interval = int(SysValid_interval)
        for well in SysValid_list:
            # for i in range(0, lc_number):
            well.append('SysQC')
        new_flat_list = []
        for i in range(0, lc_number):
            new_flat_list.append(SysValid_list[-1])
            SysValid_list = SysValid_list[:-1]
        for i in range(0, len(flattened_list), SysValid_interval):
            new_flat_list.extend(flattened_list[i:i + SysValid_interval]) #changed append() to extend()
            if i + SysValid_interval < len(flattened_list):
                for i in range(0, lc_number):
                    try:
                        if lc_number == 1:
                            if 'LIB' in new_flat_list[-1][2].upper() or 'LIB' in conditions[new_flat_list[-1][0]][0].upper():
                                try:
                                    new_flat_list.append(TB_well)
                                except NameError:
                                    raise ("TrueBlank well was not labeled in the excel sheet! It is needed because System Validation QC is," \
                                                "trying to run in the midst of the library runs.")
                        elif lc_number == 2:
                            if 'Lib' in new_flat_list[-1][2] or 'Lib' in new_flat_list[-2][2] or 'LIB' in conditions[new_flat_list[-1][0]][0].upper() or 'LIB' in conditions[new_flat_list[-2][0]][0].upper():
                                try:
                                    new_flat_list.append(TB_well)
                                    new_flat_list.append(TB_well)
                                except NameError:
                                    raise ("TrueBlank well was not labeled in the excel sheet! It is needed because System Validation QC is," \
                                                    "trying to run in the midst of the library runs.")
                        new_flat_list.append(SysValid_list[-1])
                        SysValid_list = SysValid_list[:-1]
                    except IndexError:
                        missing_SV += 1
                        continue
        for i in range(0, lc_number):
            try:
                new_flat_list.append(SysValid_list[-1])
                SysValid_list = SysValid_list[:-1]
            except IndexError:
                missing_SV += 1
                continue
        if missing_SV > 0 and SysVal_copy:
            print(f"Not enough System Validation QC was added to the plate, consider adding {missing_SV} more.")
        return new_flat_list

    def extract_file_info(self, flattened, conditions, SysValid_list, SysValid_interval, lc_number, two_xp_TB, two_xp_TB_location):
        flattened = self.insert_sysQC(flattened, SysValid_list, SysValid_interval, lc_number, two_xp_TB, two_xp_TB_location, conditions)
        well_conditions, block_runs, positions, reps, msmethods = [], [], [], [], []

        well_conditions.extend([int(w[0]) for w in flattened])
        block_runs.extend([w[2] for w in flattened])
        positions.extend([w[1] for w in flattened])
        reps = self.rep_tracker(flattened, conditions)
        msmethods.extend([conditions[int(w[0])][4] for w in flattened])
        return well_conditions, block_runs, positions, reps, msmethods

    # def two_xp_extract_file_info(flattened, conditions, SysValid_list, SysValid_interval, lc_number, two_xp_TB, two_xp_TB_location):
    #     flattened = insert_sysQC(flattened, SysValid_list, SysValid_interval, lc_number, two_xp_TB, two_xp_TB_location, conditions)
    #     well_conditions, block_runs, positions, reps, msmethods = [], [], [], [], []
    #     well_conditions.extend([int(w[0]) for w in flattened])
    #     block_runs.extend([w[2] for w in flattened])
    #     positions.extend([w[1] for w in flattened])
    #     reps = rep_tracker(flattened, conditions)
    #     msmethods.extend([conditions[int(w[0])][4] for w in flattened])
    #     return well_conditions, block_runs, positions, reps, msmethods
    
    def block(self):
        all_wells_flat = []
        two_xp_TB_location = []

        if self.cond_range1.upper() == "ALL" and self.lib_same.upper() == "YES": # one experiment, 1 or 2 column system
            for key in self.plates:
                all_wells_flat.extend(self.process_plate(self.plates[key], key, self.wet_amounts))
                two_xp_TB_location.extend(self.process_plate(self.plates[key], key, self.wet_amounts)) #, f"{two_xp_TB}-{two_xp_TB}"))
            conditions, two_xp_TB, found_TB = self.check_for_trueblank(self.all_conditions)
            if not found_TB:
                all_wells_flat.append([two_xp_TB, "R5"])
            self.compare_wells_and_counts(all_wells_flat, conditions, self.wet_amounts)
            if self.lc_number == 1:
                nonsample_before, nonsample_after, nonsample_other, column1, SysValid_list, separate_lib1 = self.column_sorter(all_wells_flat, conditions, self.wet_amounts,
                                                                                            self.num_to_run, self.lc_number, self.lib_placement, self.lib_same, self.cond_range1)
                both_blocks, num_of_blocks = self.blocker(conditions, self.even, column1)
            elif self.lc_number == 2:
                nonsample_before, nonsample_after, nonsample_other, column1, column2, extras, sysvalid_list, separate_lib2 = self.column_sorter(all_wells_flat, conditions, self.wet_amounts,
                                                                                                            self.num_to_run, self.lc_number, self.lib_placement, self.lib_same, self.cond_range1)
                both_blocks, num_of_blocks = self.blocker(conditions, self.even, column1, column2)
            # for later though # nonsample_blocks = nonsample_blocker(lc_number, nonsample_other, num_of_blocks, conditions, even)
            #raise ValueError(f'what {nonsample_other}')
            sample_blocks = self.zipper(both_blocks)
            # sample_blocks = self.block_zipper(nonsample_before, nonsample_after, sample_blocks, nonsample_blocks, self.even)
            non_flat_list = self.combine_samples_and_nonsamples(nonsample_before, nonsample_after, sample_blocks, nonsample_other, self.qc_frequency, conditions)
            # zipper to make flat list and add block labels
            flat_list = self.blocknum_and_flatten(non_flat_list)
            well_conditions, block_runs, positions, reps, msmethods = self.extract_file_info(flat_list, conditions, sysvalid_list, self.sysvalid_interval, self.lc_number, two_xp_TB, two_xp_TB_location)

        elif self.cond_range1.upper() != "ALL": # two experiments, 2 column system
            lc_number = 1
            conditions1, conditions2, two_xp_TB, found_TB = self.check_for_trueblank(self.conditions1, self.conditions2)
            all_wells_flat1 = []
            all_wells_flat2 = []
            two_xp_TB_location = []
            for key in self.plates:
                all_wells_flat1.extend(self.process_plate(self.plates[key], key, self.wet_amounts, self.cond_range1))
                all_wells_flat2.extend(self.process_plate(self.plates[key], key, self.wet_amounts, self.cond_range2))
                two_xp_TB_location.extend(self.process_plate(self.plates[key], key, self.wet_amounts, f"{two_xp_TB}-{two_xp_TB}"))
            if not found_TB:
                all_wells_flat.append([two_xp_TB, "R5"])
            # sort the wells in groups so they can be processed according to run type
            nonsample_before1, nonsample_after1, nonsample_other1, exp1col1, sysvalid_list, separate_lib1 = self.column_sorter(all_wells_flat1, conditions1,
                                                                                        self.wet_amounts, self.num_to_run, lc_number, self.lib_placement, self.lib_same, self.cond_range1)
            both_blocks1, num_of_blocks1 = self.blocker(conditions1, self.even, exp1col1)
            nonsample_before2, nonsample_after2, nonsample_other2, exp2col1, SysValid_list, separate_lib2 = self.column_sorter(all_wells_flat2, conditions2,
                                                                                        self.wet_amounts, self.num_to_run, lc_number, self.lib_placement, self.lib_same, self.cond_range1) #cond_range1 is correct, it checks of cond_range1.upper() == "ALL"
            both_blocks2, num_of_blocks2 = self.blocker(conditions2, self.even, exp2col1)

            nonsample_blocks1 = self.nonsample_blocker(lc_number, nonsample_other1, num_of_blocks1, conditions1, self.even)
            nonsample_blocks2 = self.nonsample_blocker(lc_number, nonsample_other2, num_of_blocks2, conditions2, self.even)
            sample_blocks1 = [item for block in both_blocks1 for item in block] # remove one layer of list from each list of list
            sample_blocks2 = [item for block in both_blocks2 for item in block]

            non_flat_list1 = self.combine_samples_and_nonsamples(nonsample_before1, nonsample_after1, sample_blocks1, nonsample_blocks1, self.qc_frequency, conditions1)
            non_flat_list2 = self.combine_samplaes_and_nonsamples(nonsample_before2, nonsample_after2, sample_blocks2, nonsample_blocks2, self.qc_frequency, conditions2)

            two_xp_flat_list = self.two_xp_zipper(non_flat_list1, non_flat_list2, two_xp_TB, self.all_conditions, two_xp_TB_location)
            two_xp_flat_list = self.attach_lib(two_xp_flat_list, separate_lib1, separate_lib2, two_xp_TB, two_xp_TB_location, self.lib_placement, self.lib_same)

            # adjust variable names for the rest of the code logic
            lc_number = 2
            well_conditions, block_runs, positions, reps, msmethods = self.extract_file_info(two_xp_flat_list, conditions, sysvalid_list, self.sysvalid_interval, lc_number, two_xp_TB, two_xp_TB_location)
        else:
            raise ValueError("Experiment conditions cannot be run due to missing or invalid configuration." \
            "Verify that all required experiment fields are correctly filled in the Excel sheet." \
            "If there is only one experiment the library values should be the same.")
        
        blocked = [well_conditions, block_runs, positions, reps, msmethods]
        return blocked