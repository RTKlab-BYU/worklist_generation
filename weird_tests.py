def combine_samples_and_nonsamples(sample_blocks, non_sample_blocks, QC_frequency):
    # flatten sample blocks [[[well]]]
    samples_flat = [well for block in sample_blocks for well in block]
    # assume that at this point non_sample_blocks are reformatted correctly
    counter = 0
    samples_and_non = []
    

    for well in samples_flat[:]:
        samples_and_non.append(well)
        counter += 1
        if counter == QC_frequency:
            try:
                for well in non_sample_blocks[0]:
                    samples_and_non.append(well)
            except IndexError:
                raise IndexError("Not enough QC groups were added to the plate.")
            non_sample_blocks = non_sample_blocks[1:]
            counter = 0
    return samples_and_non
        
sample_blocks = [[1, 2, 3, 4], [5, 6, 7, 8]]
non_sample_blocks = [['a', 'b'], ['c', 'd', 'e'], ['f', 'g']]
QC_frequency = 3
print(combine_samples_and_nonsamples(sample_blocks, non_sample_blocks, QC_frequency))