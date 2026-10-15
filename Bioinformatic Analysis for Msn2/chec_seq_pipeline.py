# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import chec_seq_analysis_tools as csat
from pandas import DataFrame
from time import time
from numpy import round
'''
This file will use the chec_seq_analysis_tools module  to import all the functions
needed to isolate and analyze the flanks of binding sites. Rather than using the original code,
creating a module and pulling those functions from here just makes a lot more tidy and 
presentable. Additionally, it is easier to explain what is happening at a high level and 
allows me to autotmate the process.
'''

construct = 'non_DBD'
genome_file_path = "S288C_reference_genome_R64-1-1_20110203/S288C_reference_sequence_R64-1-1_20110203.fsa"
gff_file_path = "S288C_reference_genome_R64-1-1_20110203/saccharomyces_cerevisiae_R64-1-1_20110208.gff"
counts_file_path = f"Msn2_{construct}_60sec_alignment_files_R64/Msn2_{construct}_60sec_alignment_5prime_genome_coverage.counts"
sam_file_path = f"Msn2_{construct}_60sec_alignment_files_R64/Msn2_{construct}_60sec_alignment.sam"
xmer_length = 7 # define how long the motifs of interest should be
big_xmer_length = 15 # define how big the sliding region should be pseudo-MSA
zscore = 2.5 # zscore threshold to use for defining target promoters; used the one from Kumar et al. (2023)
min_seq_size = 10 # minimum size of A-DNA seqs to look for in flanks
max_seq_size = 15 # maximum size of A-DNA seqs to look for in flanks
ts = time()
# obtain the total number of reads from the experiments
total_reads = csat.total_reads(sam_file_path) 

# map ChEC-signal to genome and normalize the reads to a certain number
chrom_length_dict, chrom_bp_arrays_dict, total_mapped_reads = csat.chec_seq_normalized_signal(counts_file_path, 1.2e7)

# use the gff file in the SGD yeast genome folder to identify coding regions
gff_data_df = csat.gff_gene_mapper(gff_file_path)

# create a dictioary to hold the sequences of each chromosome
genome_dict = csat.genome_dictionary_generator(genome_file_path)

# use data from .gff file and ChEC-seq experiments to identify and score promoter regions
promoter_data_df = csat.promoter_mapper(gff_data_df, chrom_length_dict, chrom_bp_arrays_dict)

# some promoters are bidirectional and counted twice so find all unique promoter locations
unique_promoters_df = csat.unique_promoter_locations(promoter_data_df)

# find telomeric and subtelomeric regions
cutoff_regions = csat.sub_telomeric_locator(gff_file_path)

# find the intergenic regions and save their information to a .csv file
intergenic_regions_df = csat.intergenic_region_calculator(genome_dict, gff_data_df, cutoff_regions)
intergenic_regions_df.to_csv('Msn2_Intergenic_Regions_60sec_Info.csv')

# filter out unique promoter locations that occur in telomeres and subtelomeric regions; save to .csv file
unique_nontelomeric_promoters_df = csat.unique_nonsubtelomeric_promoters(unique_promoters_df, cutoff_regions)
unique_nontelomeric_promoters_df.to_csv('Msn2_Unique_Promoters_60sec_Info.csv')

# identify all unique promoters locations with binding activity AKA nonzero signal
nonzero_signal_promoters_df = csat.nonzero_signal_promoters(unique_nontelomeric_promoters_df)

# determine target promoter locations using a user defined zscore as the threshold and save to its own .csv file
target_promoters_df = csat.target_promoters_zscore(unique_nontelomeric_promoters_df, zscore)
target_promoters_df.to_csv(f'Msn2_{construct}_60sec_Target_Promoters_Info.csv')

# create all possible xmers/motifs of a desired length and index them
# forward and reverse complements given the same index
motif_index_dict = csat.xmer_index_generator(xmer_length)

# index all bases in the genome by the xmer centered at that base
genome_index_dict = csat.genome_motif_index(genome_dict, motif_index_dict)

# score each xmer appearance in all unique promoter locations to find the most bound ones
sorted_motif_final_score_dict = csat.xmer_motif_score(unique_nontelomeric_promoters_df, chrom_bp_arrays_dict, genome_index_dict, motif_index_dict)

# idenity the 15mers with the greatest signal at each unique  
# promoter location to use for pseudo MSA later
hot_15mers = csat.hottest_big_xmers(nonzero_signal_promoters_df, chrom_bp_arrays_dict, genome_dict, xmer_length, big_xmer_length)

# also identify the 16mers with the greatest signal in all tareget promters
hot_target_15mers = csat.hottest_big_xmers(target_promoters_df, chrom_bp_arrays_dict, genome_dict, xmer_length, big_xmer_length)

# generate pfm using biopython and Motif object
pfm = csat.pfm_generator(sorted_motif_final_score_dict, hot_15mers, 30, -1)

# convert pfm to a pandas DataFrame
pfm_df = DataFrame(data=pfm.counts).transpose()

# get a DataFrame with all the information on flanking sequences
flank_coordinate_df = csat.flank_info(target_promoters_df, genome_dict, 'AGGGG', reverse_complement = True)
# filter out flank sequences that are 1 bp; need sequences with at leats 2 bp to do A-DNA analysis
flank_coordinate_df = flank_coordinate_df.loc[lambda df: df['Flank Length'] > 1, :]
flank_coordinate_df.to_csv(f'Msn2_{construct}_60sec_Target_Promoters_Flank_Coordinate_Info.csv')
tf = time()
print(f"It took {round(tf-ts,2)} seconds to run the pipeline from start to finish")
print(f'Out of {total_reads} reads from the experiment, {total_mapped_reads} were mapped giving an alignment rate of {total_mapped_reads/total_reads}')
print(pfm_df)
print(pfm.consensus)
