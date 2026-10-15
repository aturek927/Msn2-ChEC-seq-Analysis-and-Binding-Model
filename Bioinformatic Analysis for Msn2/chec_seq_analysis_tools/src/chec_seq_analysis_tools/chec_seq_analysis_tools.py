#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:45:20 2026

@author: alexturek
"""

import numpy as np
import pandas as pd
from itertools import product
from Bio import motifs
from Bio.Seq import Seq
from dna_conformation_predictors.lightgbm_predictor import AB_form_predictor, seq_to_data
from collections import Counter
from pysam import AlignmentFile

def total_reads(path_to_sam_or_bam_file):
    '''
    Function that returns the total number of reads obtained from ChEC-seq experiments
    Parameters
    ----------
    path_to_sam_or_bam_file : str
        String containing the file path to a .sam or .bam file.

    Returns
    -------
    int
        Total number of reads in .sam or .bam file which is the total number of reads obtained
        from the experiment.

    '''
    
    return AlignmentFile(path_to_sam_or_bam_file).count()

def chec_seq_normalized_signal(path_to_counts_file, normalized_reads_num):
    '''
    Function that takes in a .counts file which maps the MNase cuts and their frequency
    to the genome and the desired number of total reads at the end then outputs 2
    dictionaries containing the length and signal along each chromosome. Signal is normalized to a number of
    reads defined by the user.
    Parameters
    ----------
    path_to_counts_file : string
        file path to a .counts file containing the results analyzing ChEC-seq reads with bedtools
    normalized_reads_num : int
        number of reads to normalize the data to.

    Returns
    -------
    chrom_length_dict : dict
        dictionary containing each chromosome (key) and their lengths in base pairs (value)
    chrom_bp_arrays_dict : dict
        dictionary containing each chromosome (key) and an array containing the signal at each position in the 
        chromsome (value). Signal is normalized to normalized_reads_num/total_reads 
    total_mapped_reads : int
        the number of reads that were actually mapped to the genome
    '''
    # function that takes in a .counts file which maps the MNase cuts and their frequency
    # to the genome and the desired number of total reads at the end then outputs 2
    # dictionaries containing the length and signal along each chromosome.
    chrom_length_dict = {} # empty dictionary to hold chromosome lengths
    chrom_bp_arrays_dict = {} # empty dictionary to hold cut counts at each position
    # read .counts file into a panda DataFrame
    count_df = pd.read_csv(path_to_counts_file,sep='\t',names = ['Chromosome', 'Position', 'Count'])
    # get all the chromosome reference names and sort them alpha-numerically so they are 
    # in order by chromosome number
    ref_names = sorted(list(set(count_df['Chromosome'].values.tolist())))
    # create a dictionary that links each chromsome reference name to its number
    chrom_ref_name_dict = {ref_names[i]: f'chr {i+1}' for i in range(len(ref_names))}
    for name in ref_names:
        if chrom_ref_name_dict[name] != 'chr 17':
            count_data = count_df.loc[lambda dr: dr['Chromosome']==name,'Count'].values.tolist()
            chrom_number = chrom_ref_name_dict[name]
            chrom_length_dict[chrom_number] = len(count_data)
            chrom_bp_arrays_dict[chrom_number] = np.array(count_data)
    total_mapped_reads = sum(count_df.loc[:, 'Count'].values.tolist())
    #mito_reads = chrom_bp_arrays_dict.pop('chr 17')
    for i in chrom_bp_arrays_dict.keys():
        # in Brodsky_et_al's paper, they normalized their results to 10 million reads meaning they divided the signals at the 
        # cut sites by the total number of reads mapped then multiplied by 10 million. This allows for experiments with different
        # numbers of mapped reads to be fairly compared
        chrom_bp_arrays_dict[i]=chrom_bp_arrays_dict[i]*((normalized_reads_num)/(total_mapped_reads))
    return chrom_length_dict, chrom_bp_arrays_dict, total_mapped_reads

def chec_seq_raw_signal(path_to_counts_file):
    '''
    Function that takes in a .counts file which maps the MNase cuts and their frequency
    to the genome and the desired number of total reads at the end then outputs 2
    dictionaries containing the length and signal along each chromosome.
    Parameters
    ----------
    path_to_counts_file : string
        file path to a .counts file containing the results analyzing ChEC-seq reads with bedtools
    normalized_reads_num : int
        number of reads to normalize the data to.

    Returns
    -------
    chrom_length_dict : dict
        dictionary containing each chromosome (key) and their lengths in base pairs (value)
    chrom_bp_arrays_dict : dict
        dictionary containing each chromosome (key) and an array containing the number of cuts at each position 
        in the chromsome (value)
    total_mapped_reads : int
        the number of reads that were actually mapped to the genome
    '''
    # function that takes in a .counts file which maps the MNase cuts and their frequency
    # to the genome and the desired number of total reads at the end then outputs 2
    # dictionaries containing the length and signal along each chromosome.
    chrom_length_dict = {} # empty dictionary to hold chromosome lengths
    chrom_bp_arrays_dict = {} # empty dictionary to hold cut counts at each position
    # read .counts file into a panda DataFrame
    count_df = pd.read_csv(path_to_counts_file,sep='\t',names = ['Chromosome', 'Position', 'Count'])
    # get all the chromosome reference names and sort them alpha-numerically so they are 
    # in order by chromosome number
    ref_names = sorted(list(set(count_df['Chromosome'].values.tolist())))
    # create a dictionary that links each chromsome reference name to its number
    chrom_ref_name_dict = {ref_names[i]: f'chr {i+1}' for i in range(len(ref_names))}
    for name in ref_names:
        if chrom_ref_name_dict[name] != 'chr 17':
            count_data = count_df.loc[lambda dr: dr['Chromosome']==name,'Count'].values.tolist()
            chrom_number = chrom_ref_name_dict[name]
            chrom_length_dict[chrom_number] = len(count_data)
            chrom_bp_arrays_dict[chrom_number] = np.array(count_data)
    total_mapped_reads = sum(count_df.loc[:, 'Count'].values.tolist())
    #mito_reads = chrom_bp_arrays_dict.pop('chr 17')
    return chrom_length_dict, chrom_bp_arrays_dict, total_mapped_reads

def gff_gene_mapper(path_to_gff_file):
    '''
    Function that, when supplied with a gff file, extracts all non-dubious gene locations
    Parameters
    ----------
    path_to_gff_file : string
        File path to gff file.

    Returns
    -------
    gff_data_df : pandas DataFrame
        pandas DataFrame containing the chromosome, strand, TSS, and coding end for
        each gene in the gff file that is not labeled as "Dubious".

    '''
    # open the .gff file using with so that it closes automatically when done
    with open(path_to_gff_file, 'r') as gff:
        # for each gene, create empty list to store the neccessary info for determining promoters late
        gene_info_list = []
        # create variables that store which chromosome the program is on both as as int and as a string like in the gff file
        chrom_counter = 0
        current_chrom = 'chr0'
        for line in gff:
            # all genes and other features start with 'chr' so only look at lines that start with 'chr'
            if line.startswith('chr'):
                # the information for each feature is seaprate by tabs, so isolate each piece by splitting by tabs
                info = line.split('\t')
                # look for features labeled as genes
                if info[2] == 'gene':
                    # for a gene, get rid of the new line character in the last line of information
                    last_line = info[8].strip('\n')
                    # get the orf_class from the last line of information; need this to determine if a gene is verified
                    orf_class = last_line[last_line.index('orf'):].split('=')[1]
                    # only look at genes for which the ORF has been verified
                    if orf_class != 'Dubious':
                        # grab the gene's chromosome and strand
                        chrom = info[0]
                        strand = info[6]
                        # check if gene's chromosome matches the chromsome our program is one; change to the new if a mismatch
                        if chrom != current_chrom:
                            chrom_counter += 1
                            chrom_num = 'chr '+str(chrom_counter)
                            current_chrom = chrom
                        # gene provides left and right coordinate which are not the same as TSS and gene end depending on the
                        # strand; if forward (+) strand, TSS is left but TSS is the right coordinate if reverse (-) strand.
                        if strand == '+':
                            TSS = int(info[3])-1
                            end = int(info[4])-1
                        else:
                            TSS = int(info[4])-1
                            end = int(info[3])-1
                        # append the gene's chromosme, strand, TSS, and end to the emtpy list from earlier
                        gene_info_list.append([chrom_num, strand, TSS, end])
    # convert list of gene information into a Pandas DataFrame
    gff_data_df = pd.DataFrame(data=gene_info_list, columns=['Chromosome', 'Strand', 'TSS', 'Gene End'])
    # exclude any genes in the mitochondrial chromosome from the DataFrame
    gff_data_df = gff_data_df.loc[lambda df: df['Chromosome'] != 'chr 17',:]
    return gff_data_df

def inter_neighbor_checker(chrom_length_dict, m_chr, m_strand, m_TSS, na_chr, na_TSS, na_end, nb_chr, nb_TSS, nb_end):
    '''
    Funciton that determines the promoter length for the promoter of an input gene depending on the 
    adjacent genes. This is meant for genes in the middle of the chromosome (i.e not the first or last).
    arguements with m in front denote the current gene / gene of interest whose promoter length is
    desired. na indicates information for the gene upstream of the current one; nb indicates
    information for the gene downstream of the current one.

    Parameters
    ----------
    chrom_length_dict : dict
        Dictionary where the keys are the chromosome numbers and the values are the length of that 
        chromosome in bp.
    m_chr : string
        String indicating the chromosme of the current gene.
    m_strand : '+' or '-'
        String indicating the strand of the current gene. Forward = +, reverse = -
    m_TSS : int
        Position of the current gene's transcription start site (TSS).
    na_chr : string
        String indicating the chromosme of the gene upstream of the current one.
    na_TSS : int
        TSS for the gene upstream of the current one
    na_end :  int
        Position of the end of the gene upstream of the current one.
    nb_chr : string
        String indicating the chromosome of gene downstream of the current one.
    nb_TSS : int
        Position of the TSS for the gene downstream of the current one
    nb_end : int
        Position of the end of the gene downstream of the current one.

    Returns
    -------
    promoter_length : int
        Length of the promoter of the current gene in bp. Length of the promoter is defined as the
        the length of the sequence from the current gene's TSS to the end of the nearset adjacent gene
        region (can be on the same or different strand) or 700 bp if nearest gene's end is more than 
        700 bp from the current gene's TSS.

    '''
    # initialize variables represening the possible distances of the promoter
    # this is bassed on the distance of the current gene's TSS from the adjacent ones as well where the adjacent genes end
    test1 = na_TSS - m_TSS
    test2 = na_end - m_TSS
    test3 = nb_TSS - m_TSS
    test4 = nb_end - m_TSS
    # define the max promoter length as 700 so must look at the base with a coordinate 701 +/- from the TSS
    window_length = -1*eval(m_strand+'701')
    # define a dictionary that determine whether to take the maximum or minimum length depending on the gene's strand
    comp_dict = {'+':max , '-': min}
    # for all 3 genes on the same chromsome
    if (m_chr == na_chr) and (m_chr == nb_chr):
        # check that current TSS does not intersect either neighbor coding region
        if (test1*test2 > 0) and (test3*test4 > 0):
            lengths_list = [window_length, test1, test2, test3, test4]
            # only check lengths that are in the same direction as the promoter
            poss_lengths = [i for i in lengths_list if np.sign(i)==np.sign(window_length)]
            promoter_length = comp_dict[m_strand](poss_lengths) - np.sign(window_length)
        else:
            promoter_length = 0
    else:
        # only current and ahead genes are on the same chromosome
        # AKA current gene is first gene on its chromosome
        if m_chr == na_chr:  
            if (test1*test2 > 0):
                lengths_list = [window_length, 0-m_TSS, test1, test2]
                poss_lengths = [i for i in lengths_list if np.sign(i)==np.sign(window_length)]
                promoter_length = comp_dict[m_strand](poss_lengths) - np.sign(window_length)
            else:
                promoter_length = 0
        # only current and behind genes are on the same chromsome
        # AKA current gene is the last on its chromosome
        elif m_chr == nb_chr: 
            if (test3*test4 > 0):
                chrom_length = chrom_length_dict[m_chr]
                lengths_list = [window_length, chrom_length-m_TSS, test3, test4]
                poss_lengths = [i for i in lengths_list if np.sign(i)==np.sign(window_length)]
                promoter_length = comp_dict[m_strand](poss_lengths) - np.sign(window_length)
            else:
                promoter_length = 0
    return promoter_length

def edge_neighbor_checker(chrom_length_dict, m_chr, m_strand, m_TSS, n_chr, n_TSS, n_end, gene = 'first' or 'last'):
    '''
    Funciton that determines the promoter length for the promoter of an input gene depending on the 
    adjacent genes. This is meant for genes at the end of athe chromosome (i.e first or last).
    arguements with m in front denote the current gene / gene of interest whose promoter length is
    desired. n indicates the information for gene nearest to the current gene. It can be on the same
    strand or the other stand, it just has to be the one closest to the current gene in terms of 
    position.

    Parameters
    ----------
    chrom_length_dict : dict
        Dictionary where the keys are the chromosome numbers and the values are the length of that 
        chromosome in bp.
    m_chr : string
        String indicating the chromosme of the current gene.
    m_strand : '+' or '-'
        String indicating the strand of the current gene. Forward = +, reverse = -
    m_TSS : int
        Position of the current gene's transcription start site (TSS).
    n_chr : string
        String indicating the chromosme of the gene nearest to the current one.
    n_TSS : int
        TSS for the gene nearest the current one
    n_end :  int
        Position of the end of the gene nearest to current one.
    gene : 'First' or 'Last'
        String indicating if the current gene is the first or last one on the chromosome.


    Returns
    -------
    promoter_length : int
        Length of the promoter of the current gene in bp. Length of the promoter is defined as the
        the length of the sequence from the current gene's TSS to the end of the nearset adjacent gene
        region (can be on the same or different strand) or 700 bp if nearest gene's end is more than 
        700 bp from the current gene's TSS.

    '''
    # determine the distances between the current gene's TSS and the next gene's TSS as well as its end
    test1 = n_TSS - m_TSS
    test2 = n_end - m_TSS
    # define the max promoter length as 700 so need to look at coordinates +/- 701 from the TSS depending on the strand
    window_length = -1*eval(m_strand+'701')
    # define a dictionary that takes the max/min possible promtoer distance based on the gene's strand
    comp_dict = {'+':max , '-': min}
    # if product of the distances is negative, TSS is in the middle of a coding region so it does not have its own promoter
    if (test1*test2<0):
        promoter_length = 0
    # if product is positive, that means the TSS does not fall inside a coding region so it has its own promoter
    # depending on if the gene is the first or last on its chromsome, need to do things differently
    # start with the case of if the gene is first on the chromosome
    elif (test1*test2>0) and (gene == 'first'):
        # put all of the possible promoter lengths in a single list
        lengths_list = [window_length, 0-m_TSS, test1, test2]
        # only keep lengths that are in the same direction as the gene
        poss_lengths = [i for i in lengths_list if np.sign(i)==np.sign(window_length)]
        # take the largets/smallest length depending on the strand then account for the fact 
        # that the lengths include the TSS; depending on the strand, need to +/- 1 to the promoter length
        promoter_length = comp_dict[m_strand](poss_lengths)- np.sign(window_length)
    # address the case of the gene being the last one on the chromosome
    elif (test1*test2>0) and (gene == 'last'):
        # grab the chromosome legnth
        chrom_length = chrom_length_dict[m_chr]
        # make a list of the possible lengths of the promoter next to the last gene
        lengths_list = [window_length, chrom_length-m_TSS, test1, test2]
        # only keep lengths that are in the same direction as the gene
        poss_lengths = [i for i in lengths_list if np.sign(i)==np.sign(window_length)]
        # take the largets/smallest length depending on the strand then account for the fact 
        # that the lengths include the TSS; depending on the strand, need to +/- 1 to the promoter length
        promoter_length = comp_dict[m_strand](poss_lengths)- np.sign(window_length)
    return promoter_length

def promoter_mapper(gff_data_df, chrom_length_dict, chrom_bp_arrays_dict):
    '''
    Function that provided the gene coding regions in the form of a pandas DataFrame,
    dictionary containing the chromosome lengths, and normalized ChEC-seq signal,
    determines the location of each gene's promoter assuming it is on the same strand.

    Parameters
    ----------
    gff_data_df : pandas DataFrame
        DataFrame containing the chromsome, strand, TSS, and end coding coordinate for
        each gene from the gff file.
    chrom_length_dict : dict
        Dictionary relateing each chromosome (key) to its length in base pairs (value)
    chrom_bp_arrays_dict : dict
        Dictionary relating each chromosome (key) to the ChEC-seq signal at each of its base
        pairs as an array (value). The array is 1D so each index represents the position of
        that base on its chromsome.

    Returns
    -------
    promoter_data_df : pandas DataFrame
        DataFrame containing the chromsome, TSS, promoter end coordinate, promoter length,
        and normalized signal for each promoter. The strand of each promoter is denoted by
        the sign of the promoter length (negative length = forward strand; positive length
        = reverse strand). The normalized signal is determined by summing the signal of
        base in the promoter (which already in normalized reads), dividing by the magnitude
        of the promoter length, then multiplying by 700. 700 bp behind the TSS is the 
        cut-off used by Brodsky et al. as well as a number of other papers in the Barkai
        lab. Promoters are defined as the 700 bp upstream of a TSS or until that window
        reaches the start/end of a coding region for the nearest gene on either strand.
    '''
    
    nsites = gff_data_df.shape[0] # grab the number of sites from the DataFrame
    promoter_data_list = [] # create a list to store promoter data
    isolated_sites_counter = 0
    general_counter = 0
    for i in range(nsites): # iterate over all the rows in the DataFrame
        sitec = gff_data_df.iloc[i].values.tolist() # convert rows into lists
        chromc, strandc, TSSc, endc = sitec[0:4] # grab info for current site
        
        # for the first gene
        if i == 0:
            # grab info for site ahead
            site_na = gff_data_df.iloc[i+1].values.tolist()
            na_chr, na_TSS, na_end = site_na[0], site_na[2], site_na[3]
            # plug into edge_neighbor_checker() with gene = 'first' to get promoter length
            promoter_length = edge_neighbor_checker(chrom_length_dict,chromc,strandc,TSSc,na_chr, na_TSS, na_end, gene='first')
        # for the last gene
        elif i == (nsites-1):
            # grab info for site behind
            site_nb = gff_data_df.iloc[i-1].values.tolist()
            nb_chr, nb_TSS, nb_end = site_nb[0], site_nb[2], site_nb[3]
            # plug into edge_neighbor_checker() with gene = 'last' to get promoter length
            promoter_length = edge_neighbor_checker(chrom_length_dict,chromc,strandc,TSSc,nb_chr, nb_TSS, nb_end, gene='last')
        # for all the genes in between
        else:
            # grab info for the site ahead
            site_na = gff_data_df.iloc[i+1].values.tolist()
            na_chr, na_TSS, na_end = site_na[0], site_na[2], site_na[3]
            # grab info for the sites behind
            site_nb = gff_data_df.iloc[i-1].values.tolist()
            nb_chr, nb_TSS, nb_end = site_nb[0], site_nb[2], site_nb[3]
            # plug into inter_neighbor_checker() to get promoter length
            promoter_length = inter_neighbor_checker(chrom_length_dict,chromc, strandc, TSSc, na_chr, na_TSS, na_end, nb_chr, nb_TSS, nb_end)
        # promoter length is up to coordinate for TSS, so do not want to take TSS as the end coordinate
        if promoter_length == 0:
            signal = 0
        elif strandc == '+':
            promoter_start = TSSc - 1
            promoter_end = TSSc + promoter_length
            signal = np.sum(chrom_bp_arrays_dict[chromc][promoter_end:promoter_start+1])*(700/np.abs(promoter_length))
            isolated_sites_counter += 1
        elif strandc == '-':
            promoter_start = TSSc + 1
            promoter_end = TSSc + promoter_length
            signal = np.sum(chrom_bp_arrays_dict[chromc][promoter_start:promoter_end+1])*(700/np.abs(promoter_length))
            isolated_sites_counter += 1
        promoter_data_list.append([chromc, promoter_start, promoter_end, promoter_length,signal])
        general_counter += 1
    promoter_data_df = pd.DataFrame(data = promoter_data_list, columns = ['chr','Promoter Start Site', 'Promoter End Site', 'Promoter Length', 'Normalized Signal'])
    return promoter_data_df


def unique_promoter_locations(promoter_data_df):
    '''
    Function that filters out redundant promtoer locations from the DataFrame containing
    promoter locations. This is becuase yeast can contain bidirectional promoters where
    two adjacent genes on opposite strands may be activated by the same promoter/have their
    promoter at the same genome location. To not double count these instances, the
    coordinates of each promoter are compared to the previous one before adding it to the 
    DataFrame. Each of these bidirectional promters will have the same signal, so it is
    important to make sure they are not double counted.

    Parameters
    ----------
    promoter_data_df : pandas DataFrame
        DataFrame containing the chromsome, TSS, promoter end coordinate, promoter length,
        and normalized signal for each promoter.

    Returns
    -------
    unique_promoters_df : pandas DataFrame
        DataFrame containing the chromsome, TSS, promoter end coordinate, promoter length,
        and normalized signal for each promoter with a unique genomic location.

    '''
    # conver the DataFrame of promoter data into a list of lists
    promoter_data_list = promoter_data_df.values.tolist()
    # create an empty list to keep the data for unique promoter locations
    unique_promoter_data = []
    for i in range(len(promoter_data_list)):
        if i == 0:
            # append first line of original promoter data
            unique_promoter_data.append(promoter_data_list[i])
        else:
            # check if previous entry has the same coordinates
            chromc = promoter_data_list[i][0]
            startc = promoter_data_list[i][1]
            endc = promoter_data_list[i][2]
            chromb = promoter_data_list[i-1][0]
            startb = promoter_data_list[i-1][1]
            endb = promoter_data_list[i-1][2]
            scorec = promoter_data_list[i][4]
            scoreb = promoter_data_list[i-1][4]
            # only append the list if the current promoter has coordinates that are not the same as the promtoers
            # before and after; the promoters are organized by coordinates before this, so promoters with the same coordinates
            # will be next to each other
            if (chromc == chromb) and (startc == endb) and (endc==startb) and (scorec == scoreb):
                continue
            else:
                unique_promoter_data.append(promoter_data_list[i])
    # make a DataFrame of the unique promoter's data
    unique_promoters_df = pd.DataFrame(data = unique_promoter_data, columns = ['chr','Promoter Start Site', 'Promoter End Site', 'Promoter Length', 'Normalized Signal'] )
    return unique_promoters_df

def sub_telomeric_locator(path_to_gff_file):
    '''
    Function that determines the region inside the telomeres/telomeric regions of each chromsome. Most telomeres and telomeric
    regions tend to be heterochromatin, so they are inaccessible to transcription factors. Thus, when looking for promoters
    and other specific parts of the genome, want to limit the search to the region not containing telomerers and telomeric 
    regions.

    Parameters
    ----------
    path_to_gff_file : file
        Path to .gff file

    Returns
    -------
    cutoff_regions : dict
        Dictionary where each key is the chromosome and each value is a list containing two numbers. These mark the start
        and end coordinates for the region between telomeres/telomeric regions.
    '''
    # gff file use roman numerals instead of numbers for chromosome so create a dictionary to convert roman numerals
    # to numbers
    roman_numeral_dictionary = {'I':1, 'II':2, 'III':3, 'IV':4,
                                'V':5, 'VI':6, 'VII':7, 'VIII':8,
                                'IX':9, 'X':10, 'XI':11, 'XII':12,
                                'XIII':13, 'XIV':14, 'XV':15, 'XVI':16}
    # initialize an empty dictioanry to hold the coordinate of telomeres and telomeric features
    telo_coords_dict = {}
    # open the .gff file
    with open(path_to_gff_file, 'r') as gff:
        # initialize a variable to keep track of the current chromosome
        current_chrom = 'chr0'
        # go through the gff file line by line
        for line in gff:
            # only parse through lines that start with 'chr' as these indicate chromsome features such as genes and telomeres
            if line.startswith('chr'):
                # info about each features is tab delimitted, so split the information string by tabs
                line_info = line.split('\t')
                # check to see if the features is labeled as a telomere or repeat region, its Name or ID contains TEL, or
                # if the notes indicate that the region is part of a telomeric X element or Y' element. These are all
                # indicators that a particular feature is a in a telomere or telomeric region
                if (line_info[2] == 'telomere') or (line_info[2] == 'repeat_region') or ('Name=TEL' in line_info[8]) or ("ID=Tel" in line_info[8]) or ("Y'" in line_info[8]) or ("X%20element" in line_info[8]) :
                   # if a feature is in a telomere/telomeric region check what chromosome it is on
                    chrom = line_info[0]
                    # if the chromosome of the feature doesn't match our current chromsome, we need to create a new key
                    # and value in the dictionary to append coordinates to
                    if current_chrom != chrom:
                        # grab the roman numberal for the feature's chromosome
                        roman_numeral = chrom[3:]
                        # convert the roman numeral to a number
                        chrom_num = roman_numeral_dictionary[roman_numeral]
                        # create a key name for the feature's chromosome
                        current_chrom_name = 'chr ' + f'{chrom_num}'
                        # variable tracking the chromsome to the feature's chromosome
                        current_chrom = chrom
                        # create a new key for the feature's chromsome and set its value to an empty list
                        telo_coords_dict[current_chrom_name] = []
                        # append the feature's coordinates to emtpy list for the feature's chromosome
                        telo_coords_dict[current_chrom_name].append(int(line_info[3]))
                        telo_coords_dict[current_chrom_name].append(int(line_info[4]))
                    # if the feature's chromsome is the same as the current chromsome, just add the coordinates to that
                    # chromosome's list
                    else:
                        telo_coords_dict[current_chrom_name].append(int(line_info[3]))
                        telo_coords_dict[current_chrom_name].append(int(line_info[4]))
    # create an emtpy dictionary to hold the coordinate for the region between telomeres/telomeric regions
    cutoff_regions = {}
    # iterate over each chromosome
    for key in telo_coords_dict.keys():
        # sort the list of coordinates for the current chromosome by numerical value and convert to an array
        telo_coords_dict[key] = np.sort(telo_coords_dict[key])
        telo_coords_array = telo_coords_dict[key]
        # create two arrays representing the larger and smaller coordinates for each pair of adjacent coordinates
        # in the array
        high_coords = telo_coords_array[1:]
        low_coords = telo_coords_array[:-1]
        # compute the difference between each pair of adjacent coordinates
        coord_diff = high_coords - low_coords
        # find the index of the highest difference; this corresponds to the index for the smaller coordinate in the difference
        cutoff_low_coord_idx = np.where(coord_diff == max(coord_diff))[0][0]
        # add one to the index of the smaller coordinate to get the higher coordiante in the difference
        cutoff_high_coord_idx = cutoff_low_coord_idx+1
        # use the index values to grab the coordinates representing end of the left telomere/telomeric region and 
        #beginning of the right telomere/telomeric region
        # add 1 to the left coordinate and subract 1 from the right coordinate to get the coordinates of the region between
        # telomeres/telomeric regions
        cutoff_low_coord = telo_coords_array[cutoff_low_coord_idx]+1
        cutoff_high_coord = telo_coords_array[cutoff_high_coord_idx]-1
        # to the emptry dictionary from earlier, make each chromosmoe a key and its value a list with the start and end 
        # coordinates of the region between telomere/telomeric regions
        cutoff_regions[key] = [cutoff_low_coord, cutoff_high_coord]
    return cutoff_regions

def intergenic_region_calculator(genome_dict, gff_data_df, telomeric_regions = None):
    '''
    Function that finds the intergenic regions of the genome. Intergenic regions are defined as the regions
    between genes. This can include promoters as well as regions beyond them since so far, we have defined 
    promoters to be no more than 700 bp. In other papers, these regions have treat as a "control" to compare
    target promoters to. 

    Parameters
    ----------
    genome_dict : dictionary
        Dictionary where each key is a chromosome and their values are the chromosome sequence
    gff_data_df : Pandas DataFrame
        Pandas DataFrame containing the locations of each non-dubious gene in the genome per the .gff file from
        the SGD.
    telomeric_regions : dictionary or None (default : None), optional
        Can be None or a dictionary. The default is None. If a dictionary, must be a dictionary where each key
        is a chromosmoe whose corresponding value is a list containing the start and end coordinate of the
        region between subtelomeric regions i.e the region not occupied by telomeres and subtelomeric regions.

    Returns
    -------
    intergenic_regions_df : Pandas DataFrame
        Pandas DataFrame containing the chromosme, left coordinate, right coordinate, and sequence for each 
        intergenic region. If a dictionary containing the non-telomeric region was provided to
        telomeric_regions parameter, the intergenic regions will have been adjusted such that those completely
        in a telomeric / subtelomeric region are excluded and those that stradle the border between subtelomeric
        / telomeric and non-telomeric are adjusted such that they are not in the subtelomeric/telomeric region.
    '''
    # grab the number of genes
    num_genes = gff_data_df.shape[0]

    master_intergenic_info_list = []
    for i in range(num_genes):
        # grab information for the first gene
        current_gene = gff_data_df.iloc[i,:].values.tolist()
        # initialize variables for each genes chromosome, TSs, and gene end coordinate
        chrom_c, strand_c, tss_c, end_c = current_gene
        # convert TSS and gene end coordinates into right and left coordinates
        # for easier use
        rc_c, lc_c = max(tss_c, end_c), min(tss_c, end_c)
        if i == 0:
            # grab information for the next gene
            next_gene = gff_data_df.iloc[i+1,:].values.tolist()
            # initialize variables for each genes chromosome, TSs, and gene end coordinate
            chrom_n, strand_n, tss_n, end_n = next_gene
            # convert TSS and gene end coordinates into right and left coordinates
            # for easier use
            rc_n, lc_n = max(tss_n, end_n), min(tss_n, end_n)
            # if the gene is not at the start of the chromosome, there is an intergenic before it
            if lc_c>0:
                ig_b = [chrom_c, 0, lc_c-1, genome_dict[chrom_c][:lc_c]]
                # append info about intergenic region to master list
                master_intergenic_info_list.append(ig_b)
            # next find the intergenic region between the first and second gene
            # if both genes are on the same chromosome, region between is intergenic
            if chrom_c == chrom_n:
                # double check that there is an intergenic region between the genes
                if (lc_n>rc_c+1):
                    ig_a = [chrom_c, rc_c+1, lc_n-1, genome_dict[chrom_c][rc_c+1:lc_n]]
                else:
                    continue
            # if genes are on different chromosme, intergenic region is rest of chromosome
            # after the first gene
            else:
                ig_a = [chrom_c, rc_c+1, len(genome_dict[chrom_c])-1, genome_dict[chrom_c][rc_c+1:]]
        
        elif i == num_genes-1:
            # get coordinate for the end of the genome
            chrom_end_coord = len(genome_dict[chrom_c])-1
            # check if end of gene is also the end of the chromosome
            # no intergenic region if gene end and chromosome end are the same
            if rc_c == chrom_end_coord:
                continue
            # if gene end is not chromosome end, intergenic region is region after gene
            else:
                ig_a = [chrom_c, rc_c+1, chrom_end_coord, genome_dict[chrom_c][rc_c+1:]]
                
        else:
            # grab information for the next gene
            next_gene = gff_data_df.iloc[i+1,:].values.tolist()
            # initialize variables for each genes chromosome, TSs, and gene end coordinate
            chrom_n, strand_n, tss_n, end_n = next_gene
            # convert TSS and gene end coordinates into right and left coordinates
            # for easier use
            rc_n, lc_n = max(tss_n, end_n), min(tss_n, end_n)
            # Find the intergenic region between the current and next gene
            # if both genes are on the same chromosome, region between is intergenic
            if chrom_c == chrom_n:
                # exclude regions where genes are on different strands but overlap in coordinates
                if (rc_c+1<lc_n):
                    ig_a = [chrom_c, rc_c+1, lc_n-1, genome_dict[chrom_c][rc_c+1:lc_n]]
                else:
                    continue
            # if genes are on different chromosme, intergenic region is rest of chromosome
            # after the current gene
            else:
                ig_a = [chrom_c, rc_c+1, len(genome_dict[chrom_c])-1, genome_dict[chrom_c][rc_c+1:]]
        # append info about intergenic region to master list
        master_intergenic_info_list.append(ig_a)
    
    # define columns for intergenic regions info Pandas DataFrame
    columns = ['Chromosome', 'Left Coordinate', 'Right Coordinate', 'Sequence']
    
    # if no telomeric regions are given, put intergenic regions info into Pandas DataFrame
    if telomeric_regions == None:
        intergenic_regions_df = pd.DataFrame(master_intergenic_info_list, columns = columns)
        
    # if telomeric regions are given, fitler out intergenic regions that completely fall into them. If some
    # intergenic regions stradle between subtelomeric and non-telomeric regions, adjust them so that they are
    # completely in the non-telomeric region    
    else:
        # create an empty list to store info of passing intergenic regions
        adjusted_intergenic_info_list = []
        # for each intergenic region, check if it falls entirely or partially in a subtelomeric region
        for gene_idx, ig in enumerate(master_intergenic_info_list):
            # grab chromosome, coordinates, and sequence from info list
            chrom, lc, rc, seq = ig
            # for the gene's chromosome, grab the coordinates marking the start and end of the 
            # non-subtelomeric regions
            start, end = telomeric_regions[chrom]
            # if a intergenic region completely falls in a subtelomeric region, skip it
            if ((lc<start) & (rc<start)) or ((lc>end) & (rc>end)):
                continue
            # if one end of the intergenic region falls in a subtelomeric region, adjust
            # intergenic region's coordinates and sequence so that they don't
            # then add the adjusted intergenic region to the empty list
            elif (lc<start) & (rc>start):
                new_seq = genome_dict[chrom][start:rc+1]
                adjusted_intergenic_info_list.append([chrom, start, rc, new_seq])
            elif (lc<end) & (rc>end):
                new_seq = genome_dict[chrom][lc:end+1]
                adjusted_intergenic_info_list.append([chrom, lc, end, new_seq])
            # if an intergenic does not fall in a subtelomeric region, add it to empty list as is
            else:
                adjusted_intergenic_info_list.append(ig)
        intergenic_regions_df = pd.DataFrame(adjusted_intergenic_info_list, columns = columns)
    
    # return intergenic regions information as a Pandas DataFrame
    return intergenic_regions_df
        
def unique_nonsubtelomeric_promoters(unique_promoters_df, cutoff_regions):
    '''
    Function that filters out promoter locations that occur in telomers and subtelomeric locations. Returns a
    DataFrame of promoters not in these regions

    Parameters
    ----------
    unique_promoters_df : pandas DataFrame
        DataFrame containing the chromosome, start site, end site, length, and normalized signal of all unique
        promoter locations.
    cutoff_regions : dictionary
        Dictionary where each key is the chromosome and each value is a list where the first number is the 
        coordinate marking the end of the first telomeric region (exclude promoters from the start to this 
        coordinate) and the second number is the beginning of the second telomeric region (exclude promoters
        from the start of this coordinate to the end of the genome).
    Returns
    -------
    unique_nonsubtelomeric_promoters_df : pandas DataFrame
        DataFrame containing the chromosome, start site, end site, length, and normalized signal of all unique
        promoter locations that do not occur in a telomere or subtelomeric location.
    '''
    # create a copy of the original unique promoters DataFrame (df) so as to not change the original
    unique_promoters_df_copy  = unique_promoters_df.copy()
    # turn the rows of the copied df into a list of lists
    copy_list = unique_promoters_df_copy.values.tolist()
    # create an empty list to store the promoters that fall outside the non-telomeric region
    remove_list = []
    # check if where each promoter falls relative to the non-telomeric region
    for i in copy_list:
        # get the chrosome, promoter start, and promoter end coordinates
        chrom = i[0]
        start_site = i[1]
        end_site = i[2]
        # get the coordinates of the non-telomeric region for the same chromsome that the promoter is on
        lc_cut, rc_cut = cutoff_regions[chrom]
        # check that each end of the promoter falls inside the non-telomeric region
        if (start_site>lc_cut and end_site>lc_cut) and (start_site<rc_cut and end_site<rc_cut):
        # if both end of the promoter are in the non-telomeric region, move on to the next promoter
            continue
        else:
        # if at least one end is outside the non-telomeric region, append that promoter and its info to the
        # empty list
            remove_list.append(i)
    # remove each promoter that fell outside the non-telomeric region from the list of all promoters
    for j in remove_list:
        copy_list.remove(j)
    # return a Pandas DataFrame containing the info on all promoters that fall inside the non-telomeric region
    unique_nonsubtelomeric_promoters_df = pd.DataFrame(copy_list, columns = unique_promoters_df.columns)
    return unique_nonsubtelomeric_promoters_df
    

def nonzero_signal_promoters(unique_promoters_df):
    '''
    Function that filters out promoters with no normalized signal from the DataFrame
    containg all the promoters with unique genomic locations. This can helpful for other
    analysis down the line such with motif analysis where we want to find binding motifs
    for the transcription factor; as such , we would want to look at all promoters where
    there is a ChEC-seq signal, not all promoters in general.

    Parameters
    ----------
    unique_promoters_df : pandas DataFrame
       DataFrame containing the chromsome, TSS, promoter end coordinate, promoter length,
       and normalized signal for each promoter with a unique genomic location.

    Returns
    -------
    nonzero_signal_promoters_df : pandas DataFrame
        DataFrame containing the chromosome, TSS, promter end coordinate, promoter length,
        and normalized signal for each promoter with a unique genomic coordinate and a 
        nonzero normalized signal.

    '''
    # filter out promoter (rows) that have a normalized score of 0
    nonzero_signal_promoters_df = unique_promoters_df.loc[lambda drow: drow['Normalized Signal']>0,:]
    return nonzero_signal_promoters_df

def target_promoters_zscore(promoters_df, z_score_threshold):
    '''
    Function that uses a user defined z-score threshold to determine target promoters.
    The mean and standard deviation of the normalized scores is used to calculate the 
    z-score for each promoter and those with z-scores at or above the threshold are put into
    a new DataFrame.

    Parameters
    ----------
    promoters_df : pandas DataFrame
        DataFrame containg the chromosome, TSS, promoter end coordinate, promter length, 
        and most importantly normalized score.
    z_score_threshold : float
        z-score to use as the threshold for a target promoter.

    Returns
    -------
    target_promoters_df : pandas DataFrame
        DataFrame containg the chromosome, TSS, promoter end coordinate, promoter length,
        and normalized score for each promoter that passes the z-score threshold.
    '''
    # grab the normalized signal for each promter and put it in a list
    signal_data = promoters_df.loc[:,'Normalized Signal'].values.tolist()
    # calculate the mean and standard deviation for the set of normalized scores
    signal_mean, signal_std = np.average(signal_data), np.std(signal_data)
    # cut down the input DataFrame to only include promoters whose normalized score has a z-score
    # than the input z-score threshold
    target_promoters_df = promoters_df.loc[lambda df: ((df['Normalized Signal']-signal_mean)/signal_std)>z_score_threshold, :]
    return target_promoters_df

def tf_score_threshold(sequence_of_scores):
    # Function that determines the score thresholds for Msn2 promoters.
    # Test threshold values from 0 to 20,000 in intervals of 50.
    # Threshold value is values upon which the difference in passing sites
    # is less than 5 compared to the previous threshold.
    thresholds = np.arange(0,20050,50) # thresholds to test
    data_list = [] # list to hold threshold, # of passing sites, and passing site difference
    for i in thresholds: # iterate over thresholds
        if i == 0:
            old_score_list = list(sequence_of_scores)
            new_score_list = [j for j in old_score_list if j>i] # list of passing sites
            passing_sites = len(new_score_list) # number of passing sites
            passing_sites_diff = len(old_score_list) - passing_sites # passing site difference
            data_list.append([i, passing_sites, passing_sites_diff])# store info in list
        else:
            old_score_list = new_score_list
            new_score_list = [j for j in old_score_list if j>i] # list of passing scores
            passing_sites = len(new_score_list) # number of passing sites
            passing_sites_diff = len(old_score_list) - passing_sites # passing site difference
            data_list.append([i, passing_sites, passing_sites_diff]) # store info in list
            if passing_sites_diff<5: # if difference is less than 5, break out of loop
                break
    return i # return threshold value

def specific_and_nonspecific_reads(path_to_counts_file, target_promoter_data_file):
    '''
    Function that uses target promoter data and raw counts to figure out the number of reads that map to specific sites (AKA
    reads that map to target promoters) and nonspecific sites (AKA reads that map to everywhere else). 

    Parameters
    ----------
    path_to_counts_file : string
        file path to a .counts file containing the results analyzing ChEC-seq reads with bedtools
    target_promoter_data_file : string
        File path to .csv file containg the chromosome, TSS, promoter end coordinate, promoter length,
        and normalized score for each promoter that passes the z-score threshold.

    Returns
    -------
    num_specific_reads : int
        Number of reads that map to a target promoter.
    num_nonspecific_reads : int
        Number of reads that do not map to a target promoter.
    num_specific_sites : int
        Total number of bp on all of the target promoters. Approximate the total number of bases on target promoters as the
        the total number of specific sites.
    '''
    # from the raw counts file, a dictionaries with chromsome length, cuts at each base, and the total number of mapped reads
    chrom_length_dict, chrom_bp_arrays_dict, total_mapped_reads = chec_seq_raw_signal(path_to_counts_file)
    # load the .csv file with the target promoter data as a DataFrame
    target_promoter_data_df = pd.read_csv(target_promoter_data_file)
    # from the DataFrame, grab the location information for each promoter and put in a list of lists
    target_promoter_loc_data = target_promoter_data_df.loc[:, ['chr', 'Promoter Start Site', 'Promoter End Site']].values.tolist()
    # initializes variables to count the number of specific reads and specific sites
    num_specific_reads = 0
    num_specific_sites = 0
    # iterate over each promtoer's set of location data
    for prom in target_promoter_loc_data:
        # get the chromomsome, promoter start, and promoter end coordinate
        chrom, start, end = prom
        # convert start and end coordinates into left and right coordinates
        lc = min(start,end)
        rc = max(start,end)
        # number of specific reads is the sume of raw counts along the promoter
        # grab this from the dictionary with the cuts at each base in the genome
        num_specific_reads += np.sum(chrom_bp_arrays_dict[chrom][lc:rc+1])
        # grab the number of bases on the promoter as the number of specific sites for that promoter
        num_specific_sites += (rc-lc+1)
    # total number of nonspecific reads is the total reads subtracted by the total number of specific reads
    num_nonspecific_reads = total_mapped_reads - num_specific_reads
    # total number of nonspecific sites is the number of specific reads subtracted from the total number of bases
    num_nonspecific_sites = 12071326 - num_specific_sites
    return num_specific_reads, num_nonspecific_reads, num_specific_sites, num_nonspecific_sites

def specific_and_nonspecific_reads_exclude_telo(path_to_counts_file, target_promoter_data_file, cutoff_regions):
    '''
    Function that calculates the checks if a read maps to a target promoter or not to assign it as a specific read or not. It 
    also calculates the number of specific and nonspecific binding sites. Specific binding sites are bases in the target 
    promoters. This is subtracted from all the bases in the nontelomeric regions to get the count of nonspecific binding sites.

    Parameters
    ----------
    path_to_counts_file : string
        file path to a .counts file containing the results analyzing ChEC-seq reads with bedtools
    target_promoter_data_file : string
        File path to .csv file containg the chromosome, TSS, promoter end coordinate, promoter length,
        and normalized score for each promoter that passes the z-score threshold.
    cutoff_regions : dict
        Dictionary where each key is a chromosome and its values are the coordinates indicating the start and end of the region
        between telomeres / telomeric regions.

    Returns
    -------
    num_specific_reads : int
        Number of reads that map to a target promoter.
    num_nonspecific_reads : int
        Number of reads that do not map to a target promoter.
    num_specific_sites : int
        Total number of bp on all of the target promoters. Approximate the total number of bases on target promoters as the
        the total number of specific sites.
    num_nonspecific_sites : int
        Total number of available bases that are not part of target promoters.
    '''
    # from the raw counts file, a dictionaries with chromsome length, cuts at each base, and the total number of mapped reads
    chrom_length_dict, chrom_bp_arrays_dict, total_mapped_reads = chec_seq_raw_signal(path_to_counts_file)
    # load the .csv file with the target promoter data as a DataFrame
    target_promoter_data_df = pd.read_csv(target_promoter_data_file)
    # initialize variables to count the number of specific and nonspecific reads and sites
    num_specific_reads = 0
    num_nonspecific_reads = 0
    num_specific_sites = 0
    num_nonspecific_sites = 0
    # iterate over the chromosomes
    for chrom_num in range(1,17):
        # initialize a variable to keep track of the current chromosome as a string
        # which allows access to the dictionaries from earlier
        chrom = 'chr '+str(chrom_num)
        # grab the left and right coordinates of the region between telomeres / telomeric regions
        telo_ends = cutoff_regions[chrom]
        # from the DataFrame, grab the promoter start and end coordinates for all promoters on the current chromosome
        promoter_ends = target_promoter_data_df.loc[lambda dr: dr['chr'] == chrom, ['Promoter Start Site', 'Promoter End Site']].values.tolist()
        # calculate the total number of reads on the chromosome that are not in the telomeres and subtelomeric regions
        chrom_total_reads = chrom_bp_arrays_dict[chrom][telo_ends[0]: telo_ends[1]+1]
        # calculate the total number of bases in intergenic regions
        chrom_total_sites = telo_ends[1] = telo_ends[0] + 1
        # initialize variables to store the number of specific and nonspecific sites for each chromosome
        temp_specific_reads = 0
        temp_specific_sites = 0
        # iterate over each set of promoter coordinates
        for prom in promoter_ends:
            # initialize varialbes for the start and end coordinates of the promoter
            start = prom[0]
            end = prom[1]
            # convert the start and end coordinates to left and right coordinates
            lc = min(start,end)
            rc = max(start,end)
            # sum up the number of cuts along the promoter to get the number of specific reads
            temp_specific_reads += np.sum(chrom_bp_arrays_dict[chrom][lc:rc+1])
            # calculate the total number of bases in the promoter and take that to be the number of specific sites
            temp_specific_sites += (rc-lc+1)
        # subtract the total number of specific reads/sites from the total number of reads/sites to get the number of 
        # nonspecific reads/sites. Add all of the specific and nonspecific reads/sites to the variables at the start to keep a
        # running total of each
        num_nonspecific_reads += (chrom_total_reads - temp_specific_reads)
        num_specific_reads += temp_specific_reads
        num_nonspecific_sites += (chrom_total_sites - temp_specific_sites)
        num_specific_sites += temp_specific_sites
    return num_specific_reads, num_nonspecific_reads, num_specific_sites, num_nonspecific_sites

def specific_and_nonspecific_reads_intergenic_regions(path_to_counts_file, target_promoter_data_file, intergenic_regions_df):
    '''
    Function that calculates the checks if a read maps to a target promoter or not to assign it as a specific read or not. It 
    also calculates the number of specific and nonspecific binding sites. Specific binding sites are bases in the target 
    promoters. This is subtracted from all the bases in the intergenic regions to get the count of nonspecific binding sites.

    Parameters
    ----------
    path_to_counts_file : string
        file path to a .counts file containing the results analyzing ChEC-seq reads with bedtools
    target_promoter_data_file : string
        File path to .csv file containg the chromosome, TSS, promoter end coordinate, promoter length,
        and normalized score for each promoter that passes the z-score threshold.
    intergenic_regions_df : Pandas DataFrame
        DataFrame containing the chromosome, left and right coordinates, as well as the sequence of all intergenic regions.
        A total number of sites and reads is calulated using the intergenic regions coordinates; the number of specific 
        sites and reads is subtracted form these totals to get their nonspecific counterparts.

    Returns
    -------
    num_specific_reads : int
        Number of reads that map to a target promoter.
    num_nonspecific_reads : int
        Number of reads that do not map to a target promoter.
    num_specific_sites : int
        Total number of bp on all of the target promoters. Approximate the total number of bases on target promoters as the
        the total number of specific sites.
    num_nonspecific_sites : int
        Total number of available bases that are not part of target promoters.
    '''
    # from the raw counts file, a dictionaries with chromsome length, cuts at each base, and the total number of mapped reads
    chrom_length_dict, chrom_bp_arrays_dict, total_mapped_reads = chec_seq_raw_signal(path_to_counts_file)
    # load the .csv file with the target promoter data as a DataFrame
    target_promoter_data_df = pd.read_csv(target_promoter_data_file)
    # initialize variables to count the number of specific and nonspecific reads and sites
    num_specific_reads = 0
    num_nonspecific_reads = 0
    num_specific_sites = 0
    num_nonspecific_sites = 0
    # iterate over the chromosomes
    for chrom_num in range(1,17):
        # initialize a variable to keep track of the current chromosome as a string
        # which allows access to the dictionaries from earlier
        chrom = 'chr '+str(chrom_num)
        # grab the information for the intergenic regions of the current chromosome
        chrom_intergenic_regions = intergenic_regions_df.loc[lambda df: df['Chromosome'] == chrom, :]
        # get the number of intergenic regions for this chromosome
        num_chrom_intergenic_regions = chrom_intergenic_regions.shape[0]
        # put the left and right coordinates of each intergenic region into an array
        chrom_intergenic_coordinates = chrom_intergenic_regions.loc[:,['Left Coordinate', 'Right Coordinate']].values
        # calculate the length of each intergenic region from its coordinates
        chrom_intergenic_lengths = chrom_intergenic_coordinates[:,1] - chrom_intergenic_coordinates[:,0] + 1
        # from the DataFrame, grab the promoter start and end coordinates for all promoters on the current chromosome
        promoter_ends = target_promoter_data_df.loc[lambda dr: dr['chr'] == chrom, ['Promoter Start Site', 'Promoter End Site']].values.tolist()
        # calculate the total number of reads on the chromosome that are not in the telomeres and subtelomeric regions
        chrom_total_reads = sum([sum(chrom_bp_arrays_dict[chrom][chrom_intergenic_coordinates[i,0]:chrom_intergenic_coordinates[i,1]]) for i in range(num_chrom_intergenic_regions)])
        # calculate the total number of bases in intergenic regions
        chrom_total_sites = np.sum(chrom_intergenic_lengths)
        # initialize variables to store the number of specific and nonspecific sites for each chromosome
        temp_specific_reads = 0
        temp_specific_sites = 0
        # iterate over each set of promoter coordinates
        for prom in promoter_ends:
            # initialize varialbes for the start and end coordinates of the promoter
            start = prom[0]
            end = prom[1]
            # convert the start and end coordinates to left and right coordinates
            lc = min(start,end)
            rc = max(start,end)
            # sum up the number of cuts along the promoter to get the number of specific reads
            temp_specific_reads += np.sum(chrom_bp_arrays_dict[chrom][lc:rc+1])
            # calculate the total number of bases in the promoter and take that to be the number of specific sites
            temp_specific_sites += (rc-lc+1)
        # subtract the total number of specific reads/sites from the total number of reads/sites to get the number of 
        # nonspecific reads/sites. Add all of the specific and nonspecific reads/sites to the variables at the start to keep a
        # running total of each
        num_nonspecific_reads += (chrom_total_reads - temp_specific_reads)
        num_specific_reads += temp_specific_reads
        num_nonspecific_sites += (chrom_total_sites - temp_specific_sites)
        num_specific_sites += temp_specific_sites
    return num_specific_reads, num_nonspecific_reads, num_specific_sites, num_nonspecific_sites
    

def seq_reverser(sequence):
    '''
    Function that takes a DNA sequence and outputs its reverse complement which inverts the sequence and swaps each base for 
    its parnter

    Parameters
    ----------
    sequence : string
        Sequence whose reverse complement is to be determined

    Returns
    -------
    reverse_complement : string
        Reverse complment of the DNA sequence input into the function
    '''
    # create a dictionary where key is a DNA basepair and its value is its complementary base pair
    base_dict = {'A':'T', 'C':'G', 'T':'A', 'G':'C'}
    # iterate over each character's index in the input sequence
    for i in range(len(sequence)):
        # for the first base, create a variable that swaps the first base of the input sequece for its complementary base
        if i == 0:
            complement = base_dict[sequence[i]]
        # for each base after the first one, add its complement to the variable created for the first base
        else:
            complement += base_dict[sequence[i]]
    # reverse the complmentary sequence
    reverse_complement = complement[::-1]
    return reverse_complement

def xmer_index_generator(xmer_length):
    '''
    Function that generates all possible xmers with the user defined length. It then assigns
    each one an index starting from one and xmers that are reverse complement are given the
    same index

    Parameters
    ----------
    xmer_length : int
        Desired xmer length.

    Returns
    -------
    motif_index_dict : dict
        Dictionary containing each xmer, both forward and reverse complement, as the key
        and their index as their value.
    '''
    motif_index_dict = {} # empty dictionary to hold motifs and their indicies
    split_motifs = list(product('AGTC', repeat=xmer_length)) # generate all possible xmers
    joined_motifs = list(map(lambda x: ''.join(x), split_motifs)) # convert 7mer tuples into strings
    motif_counter = 1 # keep track of motif indices
    # assign indices to each motif with complments having the same index
    for motif in joined_motifs:
        # check if complement is already in motif index dictionary 
        if seq_reverser(motif) in motif_index_dict.keys():
            # if complement is there, assign same index as complement
            motif_index_dict[motif] = motif_index_dict[seq_reverser(motif)]
        else:
            # if complement is not in there, assign index directly to it and increase counter
            motif_index_dict[motif] = motif_counter
            motif_counter += 1
    return motif_index_dict

def genome_dictionary_generator(path_to_genome_file):
    '''
    Function that generates a dictionary relating each chromosome (key) to its forward
    sequence when provided with a genome in the .fsa file format.

    Parameters
    ----------
    path_to_genome_file : string
        File path to genome file. It should be in the fastq format (.fsa extension) as the
        function is specifically designed to parse that kind of file.

    Returns
    -------
    genome_dict : dict
        Dictionary where each chrosome is a key and its actual sequences from the genome
        file is its value.
    '''
    # read genome file
    with open(path_to_genome_file, 'r') as f:
        chrom_counter = 1 # keep track of the number of chromosomes
        genome_dict = {} # empty dictionary to hold genome
        for line in f:
            # first line of each chromosome has metadata, not sequence, so check first
            if line.startswith('>ref'):
                # if its the first line, take the opportunity to assign chromosome variable
                chrom = 'chr '+str(chrom_counter)
                chrom_counter += 1 # increase chromosome counter for next chromosome
            else:
                # check if line is the first line containing the chromosome sequence
                if chrom not in genome_dict.keys():
                    # if its the first sequenc line, initialize dictionary key and value
                    genome_dict[chrom] = line.strip('\n')
                else:
                    # if it not the first sequence line, add sequence to current one
                    genome_dict[chrom] += line.strip('\n')
    # get rid of mitochondria chromosome (chromosome 17)
    del genome_dict['chr 17']
    return genome_dict

def genome_base_freqs(genome_dict):
    '''
    Function that calculate the frequency of each base in a genome when the genome is
    provided as a dictionary containing each chromosome (key) and their sequence (value).
    This frequency is for both strands, not just a single strand; the frequencies of A's and
    T's are the same while those for G and C are the same.

    Parameters
    ----------
    genome_dict : dict
        Dictionary containg the chromosomes of the genome (key) and their sequences 
        (values).

    Returns
    -------
    A_prob : float
        Probability of finding an adenosine anywhere in the genome.
    G_prob : float
        Probability of finding a guanine anywhere in the genome.
    T_prob : float
        Probability of finding a thymine anywhere in the genome.
    C_prob : float
        Probability of finding a cytosine anywhere in the genome.
    '''
    # intialize variables to store the frequency of each base pair
    A_freq, G_freq, T_freq, C_freq = np.zeros(4)
    # iterate over each chromosome sequence
    for chrom_seqs in genome_dict.values():
        # add the frequency of each base in the chromosome to the variables intiailize earlier
        # determine frequency of duplex DNA, not just the single forward strand
        A_freq += chrom_seqs.count('A') + chrom_seqs.count('T')
        G_freq += chrom_seqs.count('G') + chrom_seqs.count('C')
        T_freq += chrom_seqs.count('T') + chrom_seqs.count('A')
        C_freq += chrom_seqs.count('C') + chrom_seqs.count('G')
    # put all of the basepair frequencies into an array
    base_freqs = np.array([A_freq, G_freq, T_freq, C_freq])
    # calculate the total number of bases
    total_bases = sum(base_freqs)
    # get the probability of each base in the duplex by dividing the frequencies by the total number of bases
    base_probs = base_freqs/total_bases
    # initialize a variable for each base probability
    A_prob, G_prob, T_prob, C_prob = base_probs
    return [A_prob, G_prob, T_prob, C_prob]

def genome_motif_index(genome_dict, motif_index_dict):
    '''
    Function that, using the index outline in the dictionary created by the
    motif_generator_function(), provdies an index to almost every base in the genome
    contained in genome_dict. The index is determined by the xmer that the base is centered
    on.
    Parameters
    ----------
    genome_dict : dict
        Dictionary containing each chromosome (key) and their sequence (value) for a 
        genome.
    motif_index_dict : dictioary
        Dictionary containing each possible xmer of a certain length, both forward and 
        reverse complement, where forward and reverse complements have the same index.

    Returns
    -------
    genome_index_dict : dict
        Dictionary containing the chromosome of each genome (key) and an 1D array where each
        position represents the location of a base on that chromsome and the value at the 
        position is the index of the xmer centered at that location.
    '''
    genome_index_dict = {} # empty dictionary to hold xmer indicies for each nucleotide
    # grab the first motif in motif dictionary to deterine xmer_length
    xmer_length = len(list(motif_index_dict.keys())[0])
    # calculate offset to start indexing genome from (lower_half) and cut-off position to stop indexing 
    # genome at (upper_half)
    lower_half = xmer_length // 2
    upper_half = xmer_length - lower_half - 1
    for chromosome in genome_dict.keys(): 
        genome_index_dict[chromosome] = np.zeros(len(genome_dict[chromosome]))
        for pos in range(lower_half,len(genome_dict[chromosome])-upper_half): 
            genome_index_dict[chromosome][pos] = motif_index_dict[genome_dict[chromosome][pos-lower_half:pos+upper_half+1]]
    return genome_index_dict

def xmer_motif_score(unique_promoters_df, chrom_bp_arrays_dict, genome_index_dict, motif_index_dict):
    '''
    Function that calculates the relative average binding scores for each xmer of a length
    set by the function motif_index_generator() where the user inputs an xmer length. This
    score is calculated by summing the ChEC-seq signal in a window outside the xmer, summing
    the ChEC-seq signal inside the xmer, averaging both scores by their length, then taking
    the differrence between them (outer - inner). This is the same protocol used by Kumar
    et al. (2023) with the idea being that at bound xmers, the signal is high outside of
    the xmer but low inside the xmer since the MNase cuts outside where the TF binds. Thus,
    higher scoring xmers should represent xmers that are more likely to be bound. With Msn2,
    the motif is known to be AGGGG which is 5 bp. As is done in the paper, the inner score 
    is the sum of cuts for along each xmer occurence while the outer score is the sum of cuts
    along the 31 bp centered on the xmer (+/- 15 bp from the center of the xmer).

    Parameters
    ----------
    unique_promoters_df : dataFrame
        DataFrame containing the chromosome, TSS, promoter end coordinate, promoter length, 
        and normalized score of each promote with a unique genomic location.
    chrom_bp_arrays_dict : dict
        Dictionary where each chromosome (key) is related to a 1D array (value) where the 
        position in the array represents the same position on the chromosome and the value 
        is the normalized ChEC-seq signal at that position.
    genome_index_dict : dict
        Dictionary containing the chromosome of each genome (key) and an 1D array where each
        position represents the location of a base on that chromsome and the value at the 
        position is the index of the xmer centered at that location
    motif_index_dict : dict
        Dictionary containing each possible xmer of a certain length, both forward and 
        reverse complement, where forward and reverse complements have the same index.

    Returns
    -------
    sorted_motif_final_score_dict : dict
        Dictionary where each key is a string containing an xmer (both the forward and 
        reverse complement sequences) and its values is its relative binding score. The 
        dictionary is sorted by relative binding score meaning the highest bound xmers
        are at the beginning.

    '''
    # create a dictionary where each key is an xmer (forward and reverse) and its value is its binding score
    motif_index_score_dict={}
    # grab the length of the xmer
    xmer_length = len(list(motif_index_dict.keys())[0])
    # calculate offset to start indexing genome from (lower_half) and cut-off position to stop indexing 
    # genome at (upper_half)
    lower_half = xmer_length // 2
    upper_half = xmer_length - lower_half - 1
    # iterate over each promoter in the DataFrame
    for row in range(unique_promoters_df.shape[0]):
        # grab the information for the promoter
        info = unique_promoters_df.iloc[row,:].values.tolist()
        # grab the promoter's chromosome from its information in the DataFrame
        chrom = info[0]
        # convert the promoter's start and end coordinate into left and right coordinates
        promoter_lc = min(info[1], info[2])
        promoter_rc = max(info[1], info[2])
        # iterate over each base in the promoter but start at the 15th base and stop 15 bases from the end
        for pos in range(promoter_lc+15, promoter_rc-14):
            # calculate the average score of the outer window (+/- 15 bp from the center of the xmer)
            outer_window_score = np.sum(chrom_bp_arrays_dict[chrom][pos-15:pos+16])/31
            # calculate the score of the inner window (score along the xmer)
            inner_window_score = np.sum(chrom_bp_arrays_dict[chrom][pos-lower_half:pos+upper_half+1])/xmer_length
            # subract the inner score from the outer score
            score = outer_window_score - inner_window_score
            # If the xmer's genome index is not already in the dictionary created earlier, add it to the dictionary
            if genome_index_dict[chrom][pos] not in motif_index_score_dict.keys():
                motif_index_score_dict[genome_index_dict[chrom][pos]] = np.zeros(2)
            # for each xmer, keep a running total of its score along as well as its nubmer of occurences
            motif_index_score_dict[genome_index_dict[chrom][pos]][0] += score
            motif_index_score_dict[genome_index_dict[chrom][pos]][1] += 1
    # create an empty dictionary for storing the relative binding score of each motif
    motif_final_score_dict = {}
    # iterate over each motif index in the dictionary with each motif's total score and occurences
    for motif_index in motif_index_score_dict.keys():
        # for each motif, grab its forward and reverse sequence
        pos_motifs = list(filter(lambda key: motif_index_dict[key] == motif_index, motif_index_dict))
        # score each motif by normalizing each motif's score to its total number of occurences (AKA relative binding score)
        # store these scores along with the motif sequences in the emtpy dictioary created earlier
        motif_final_score_dict[pos_motifs[0]+'/'+pos_motifs[1]] = motif_index_score_dict[motif_index][0]/motif_index_score_dict[motif_index][1]
    # sort the dictionary with relativen binding scores such that the highest scores are at the top / shown first
    sorted_motif_final_score_dict = {k:v/max(motif_final_score_dict.values()) for k,v in sorted(motif_final_score_dict.items(), key = lambda item: item[1], reverse=True)}
    return sorted_motif_final_score_dict

def hottest_big_xmers(promoters_df, chrom_bp_arrays_dict, genome_dict, xmer_length, big_xmer_length):
    '''
    Function that extracts xmers of a user defined length (called big xmers) from the
    promoters defined in the DataFrame provided by the user. These big xmers should have a
    length longer than the xmers that represent possible binding motifs. The idea here is to
    extract big xmers and use them them later for a pseudo-MSA.

    Parameters
    ----------
    promoters_df : DataFrame
        DataFrame containing the chromosome, TSS, end promoter coordinate promoter length, 
        and normalized score for the promoters of interest.
    chrom_bp_arrays_dict : dict
        Dictionary where each chromosome (key) is related to a 1D array (value) where the 
        position in the array represents the same position on the chromosome and the value 
        is the normalized ChEC-seq signal at that position.
    genome_dict : dict
        DESCRIPTION.
    xmer_length : int
        Length of xmers that are tested to see if they are binding motifs.
    big_xmer_length : int
        Length of big xmers; this must be bigger than than xmer_length by at least 2.
    Returns
    -------
    hot_promoter_big_xmers : list
        List containing the sequences of big xmers with the highest score for each 
        promoter. Big xmers are extracted by finding the xmer with the highest score
        then extracting the big xmer with big_xmer_length centered on that highest scoring
        xmer. Xmers are scored in the way as when they are analyzed to see which one is
        the most bound.

    '''
    # create an empty list for storing the hottest xmers of each promoter
    hot_promoter_big_xmers = []
    # calculate start position for xmer (lower_half) and end position (upper_half)
    big_lower_half = big_xmer_length // 2
    big_upper_half = big_xmer_length - big_lower_half - 1
    # calculate start position for xmer (lower_half) and end position (upper_half)
    lower_half = xmer_length // 2
    upper_half = xmer_length - lower_half - 1
    # iterate over each promoter(AKA row) in the input DataFrame
    for row in range(promoters_df.shape[0]):
        # grab the promoter information for the current promoter
        info = promoters_df.iloc[row,:].values.tolist()
        # grab the promoter's chromosome from the information
        chrom = info[0]
        # grab the promoter's start and end coordinate and covert them to left and right coordinates
        promoter_lc = min(info[1], info[2])
        promoter_rc = max(info[1], info[2])
        # initilaize a variable to keep track of the score for for the "hottest" xmer
        best_promoter_score = 0
        # iterate over each base in the promoter but start at the 15th base and stop 15 bases from the end
        for pos in range(promoter_lc+15, promoter_rc-14):
            # calculate the outer window score (+/-15 bp on the center of the xmer)
            outer_window_score = np.sum(chrom_bp_arrays_dict[chrom][pos-15:pos+16])/31
            # calculate the inner window score of the xmer (just along the xmer)
            inner_window_score = np.sum(chrom_bp_arrays_dict[chrom][pos-lower_half:pos+upper_half+1])/xmer_length
            # get the overall score by subtracting the inner score from the outer score
            score = outer_window_score - inner_window_score
            # if the current xmer score is greater than the previous hottest xmer score, replace it
            # also createa a variable "hot_xmer" that holds the sequence of said xmer
            if score > best_promoter_score:
                best_promoter_score = score
                hot_xmer = genome_dict[chrom][pos-big_lower_half:pos+big_upper_half+1]
        # append the hot_xmer to the emtpy list from earlier
        hot_promoter_big_xmers.append(hot_xmer)
    return hot_promoter_big_xmers

def pfm_generator(sorted_motif_final_score_dict, hot_promoter_big_xmers, num_motifs_scored, anchor_motif_num):
    '''
    Function that generates a position frequency matrix (PFM) from the xmer motif scores and 
    highest scoring big xmers provided by the user. The highest scoring xmer is used as an
    anchor upon which all other motifs are compared and given an alignment score. However, 
    motifs are chosen by sliding along each of the highest scoring big xmers and picking 
    the xmer within the big xmer that has the highest alignment score. The user can the
    use the num_motifs_scored parameter to specify how many of the selected xmers are used
    for creating the PFM. THe selected xmers are sorted before the num_motifs_scored 
    parameters is used to cut selected xmers meaning a smaller values equates to the 
    selected xmers that best align to the anchor are used to create the PFM.

    Parameters
    ----------
    sorted_motif_final_score_dict : dict
        sorted_motif_final_score_dict : dict
            Dictionary where each key is a string containing an xmer (both the forward and 
            reverse complement sequences) and its values is its relative binding score. The 
            dictionary is sorted by relative binding score meaning the highest bound xmers
            are at the beginning.
    hot_promoter_big_xmers : list
        List containing the sequences of big xmers with the highest score for each 
        promoter. Big xmers are extracted by finding the xmer with the highest score
        then extracting the big xmer with big_xmer_length centered on that highest scoring
        xmer. Xmers are scored in the way as when they are analyzed to see which one is
        the most bound.
    num_motifs_scored : int
        Number of top motifs to use when constructing PFM.
    anchor_motif_num : int; 1 or -1 
        Indicates which orientation of the motif to use as the anchor motiof. Pick 1 to use 
        the forward orientaiton or -1 to use the reverse complement. The results should be
        roughly the same if the consensus.

    Returns
    -------
    pfm : Motif
        Motif object from the biopython library. It contains a frequency matrix
        displaying the frequency of each base at each position which depend on the motifs
        used for creating the PFM. The Motif object has other attributes that can be used
        to get more information from the PFM or even transform it. It can also be feed into
        a pandas DataFrame (pd.DataFrame(data=Motif).transpose() to get it in same 
        orientation as the Motif object).

    '''
    # keys/sequences in the dictionary contain the forward and revsere sequence seaparated by '/', so split the string by '/'
    best_motif = list(sorted_motif_final_score_dict.keys())[0].split('/')
    # store the forward and reverse sequences of the highest scoring motif in two variables
    best_seq1 = best_motif[0]
    best_seq2 = best_motif[1]
    # create a dictionary which assigns the forward motif +1 and -1 to the reverse motif
    seq_dict = {1: best_seq1, -1: best_seq2}
    # choose which motif to align to based on the anchor_motif_num given by the user (+1 for forward and -1 for reverse)
    anchor_motif = seq_dict[anchor_motif_num]
    # get the length of the motif
    xmer_length = len(best_seq1)
    # create an empty dictionary to store the best aligned motif of each hottest big xmers 
    best_aligned_motifs_dict = {}
    # iterate over each of the 'hottest' xmers in the each promoters
    for seq in hot_promoter_big_xmers:
        # for a hot xmer, make a list with every subsequence within it that is the same size as the motif
        submotifs = [seq[start:start+xmer_length] for start in range(len(seq)-xmer_length+1)]
        # to list of subsequences, add each subsequenceds reverse complement
        submotifs_all = submotifs + [seq_reverser(mot) for mot in submotifs]
        # intialize a variable to keep track of the highest scoring subsequence
        best_score = 0
        # iterate over each subsequence, forward and reverse
        for mot in submotifs_all:
            # initialize another variable to keep track of track of the current subsequence's score
            score = 0
            # score the subsequence based on how well it aligns with the anchor motif by giving one point for each matching base
            # max score is the length of the motif
            for base_pos in range(len(mot)):
               if mot[base_pos] == anchor_motif[base_pos]:
                   score += 1
            # check if the current subsequence scores better than the previous highest scoring motif
            # if so, change the variables holding the highest scoring motif and its score
            if score > best_score:
                best_score = score
                best_motif = mot
        # store the best scoring/most aligned motif of that xmer in the empty dictionary from earlier
        # where each key is the best scoring motif and each value is the score
        best_aligned_motifs_dict[best_motif] = best_score
    # sort the dictionary of aligned motifs by their score such that the highest scoring ones are first/at the top
    sorted_best_aligned_motifs_dict = {k:v for k,v in sorted(best_aligned_motifs_dict.items(), key = lambda x: x[1], reverse = True)}
    # create a list that only includes the x amount of highest scoring motifs where x = num_motifs_scored
    best_aligned_motifs = list(sorted_best_aligned_motifs_dict.keys())[:num_motifs_scored]
    # with a list of the top sequences that best match the anchoring top 1,
    # create a PWM using the relative binding scores; weight frequency
    # of bases in the motifs by the score of the motifs they appear in
    
    # for the x number of best aligning motifs, convert them into sequence objects
    top_xmers = [Seq(k) for k in best_aligned_motifs]
    # pass the list of sequences objects to PWM toolf from the Biopython module
    pfm = motifs.create(top_xmers)
    return pfm

def log_odds_motif_score(sequence, log_odds_array):
    # function that given a sequence and Log Odds matrix made from Msn2 reads,
    # scores the sequence. A higher score means that the sequence is more likely to be 
    # bound while a lower score means it is less likely to be bound.
    base_dict = {'A':0, 'T':1, 'G':2, 'C':3}
    score = 0
    for i in range(len(sequence)):
        score += log_odds_array[base_dict[sequence[i]],i]
    return score

def flank_finder(promoter_seq, motif, reverse_complement = True):
    '''
    Function that, when given a sequence(promoter_seq) and motif, finds each occurence of the motif in the sequence then
    returns the start and end coordinates of each motif in a list as well as list of all the sequences flanking the motifs. If
    desired, the user can determine whether to look for the motif as or also look for occurences of the motif's 
    reverse comliment

    Parameters
    ----------
    promoter_seq : string
        Sequence, typically of a promoter, to check for motif occurences.
    motif : string
        Motif to look for in the input sequence (promoter_seq).
    reverse_complement : Boolean, optional
        When True, the function will look for occurences of the motif and its revesre complement. When set to False, it will
        only look for occurences of the motif as it suplied. The default is True.

    Returns
    -------
    motif_index_list : list
        List containing the start coordinates of each motif occurence in the sequence. These are relative to the
        sequence itself.
    flank_list : List
        List containing strings that are all the sequences around each motif occurence. 
    '''
    # function works differently if reverse_complemnt is True or False; start with the case if it True
    if reverse_complement == True:
        # determine the reverse complment of the motif
        rev_motif = seq_reverser(motif)
        # count the number of times the motif (forward and reverse) occurs in the sequence
        motif_count = promoter_seq.count(motif) + promoter_seq.count(rev_motif)
        # find the length of the motif
        mot_len = len(motif)
        # create empty list to hold the flanking sequences and motif coordinates
        flank_list = []
        motif_index_list = []
        # initialize a variable representing the rest of the sequence ahead after looking at a motif occurence
        test_seq = promoter_seq
        # iterate over the number of times the motif appears
        for count in range(motif_count):
            # need do things slightly different for the first motif occurence
            if count == 0:
                # get the starting coordinates of the first forard and reverse motif occurences
                for_motif_index = test_seq.find(motif)
                rev_motif_index = test_seq.find(rev_motif)
                # if the overall sequence has both, start with the one that occurs first
                # meaning the one with the smaller coordinate
                if (for_motif_index > -1) and (rev_motif_index > -1):
                    motif_index = min(for_motif_index, rev_motif_index)
                # if only one is present in the entire sequence, one of the coordinates will be -1 so 
                # just need to take the one that is not -1
                else:
                    motif_index = max(for_motif_index, rev_motif_index)
                # append the starting coordinate of the motif to one of the empty lists from earlier
                motif_index_list.append(motif_index)
                # update the test_seq variable to only include the rest of the sequence after this first motif occurnce
                test_seq = promoter_seq[motif_index+mot_len:]
                # if the motif does not occur at the beginning of the sequence, the sequence behind the motif is a flank
                # so need to append that sequence to the other emtpy list
                if motif_index != 0:
                    flank_list.append(promoter_seq[0:motif_index])
            # for all motif occurences after the first one
            else:
                # get the starting coordinates of the next occurence of the forward and reverse motif
                for_motif_index = test_seq.find(motif)
                rev_motif_index = test_seq.find(rev_motif)
                # if remainig sequnce has both, choose the one that occurs (smaller coordinate)
                if (for_motif_index > -1) and (rev_motif_index > -1):
                    rel_motif_index = min(for_motif_index, rev_motif_index)
                else:
                    rel_motif_index = max(for_motif_index, rev_motif_index)
                # because we used the rest of the sequence after the previous motif occurence,
                # nee to adjust the starting coordinate so that it is relative to the entire sequence
                motif_index = motif_index_list[count-1]+mot_len+rel_motif_index
                # append the motif starting coordinate to the 
                motif_index_list.append(motif_index)
                # adjust the variable representing the rest of the sequence after the current motif occurence
                test_seq = promoter_seq[motif_index+mot_len:]
                # if the current motif is not right next to the previous one, then the sequence in between motifs
                # is a flank and needs to be appended to the list of flank sequences
                if motif_index_list[count] != motif_index_list[count-1]+mot_len:
                    flank_list.append(promoter_seq[motif_index_list[count-1]+mot_len:motif_index])
        # after the last motif occurence, if there is still sequence after it, it is a flank that needs to be added ot the list
        if test_seq != '':
            flank_list.append(test_seq)
    # code for reverse_complement = False works the same as the True case, just without needing to decide if a motif
    # occurence is the forward or reverse
    else:
        motif_count = promoter_seq.count(motif) 
        mot_len = len(motif)
        flank_list = []
        motif_index_list = []
        test_seq = promoter_seq
        for count in range(motif_count):
            if count == 0:
                motif_index = test_seq.index(motif)
                motif_index_list.append(motif_index)
                test_seq = promoter_seq[motif_index+mot_len:]
                if motif_index != 0:
                    flank_list.append(promoter_seq[0:motif_index])
            else:
                motif_index = motif_index_list[count-1]+mot_len+test_seq.index(motif)
                motif_index_list.append(motif_index)
                if motif_index_list[count] != motif_index_list[count-1]+mot_len:
                    flank_list.append(promoter_seq[motif_index_list[count-1]+mot_len:motif_index])
        test_seq = promoter_seq[motif_index+mot_len:]
        if test_seq != '':
            flank_list.append(test_seq)
    return motif_index_list, flank_list

def flank_position_finder(promoter_seq, motif_index_list, motif):
    '''
    Function that when provided with the promoter sequence, list containing the motif
    starting positions, and the motif itself, will output a list containing the starting and
    ending positions of each flank. The list of motif positions should be relative to the 
    promoter (position 0 means the start of the promoter sequence, not the start of the 
    chromosome), so the flank positions will be relative to the promoter sequence as well. This is to
    be used following the flank_finder() function and using the motif positions from that. As a result,
    whether this function also maps the flank positions for the reverse motif entirely depends on if
    the flank_finder() function had the reverse_complement parameter equal to True or False.

    Parameters
    ----------
    promoter_seq : str
        String representation of the promoter sequence.
    motif_index_list : iter
        Itearble containing ints that represent the starting positions of the motif.
    motif : str
        String representation of the motif.

    Returns
    -------
    flank_pos_list : list
        List containing the starting and ending positions of each flank. The list is
        organized as follows: [flank 1 start, flank 1 end, flank 2 start, flank 2 end, 
        etc.].
    '''
    # grab the length of the input motif
    mot_len = len(motif)
    # create an emtpy list to store the position of the start and end of each motif in the promoters
    flank_pos_list = []
    # iterate over each motif start coordinate's index
    for x in range(len(motif_index_list)+1):
        # for the first motif occurence, need to check if if the motif starts that the beginning of the sequence
        if x == 0:
            # if the does not start at the beginning of the sequence, everything before it is a flank sequence
            if motif_index_list[x] != 0:
                # append the start of the flank (0) and end of the flank (motif start coordinate - 1) to the empty list
                flank_pos_list.append(0)
                flank_pos_list.append(motif_index_list[x]-1)
            # if the motif is at the start of the sequence, no flank behind it so move to the next motif occurence
            else:
                continue
        # for the last motif occurence, need to check if there is a flank ahead of it as well
        elif x == len(motif_index_list):
            # if end of the promoter ends when the last motif occurence ends, don't do anything
            if motif_index_list[x-1]+mot_len == len(promoter_seq):
                continue
            else:
                # if there is still sequence after the last motif occurence, it is a flank so append its coordinates to the list
                flank_pos_list.append(motif_index_list[x-1]+mot_len)
                flank_pos_list.append(len(promoter_seq)-1)
        # for all other other motif occurences, there is flank behind it (including the last one) if the motif before does
        # not occure directly behind it
        else:
            # check that the current motif is not right next to the previous motif
            if motif_index_list[x-1]+mot_len != motif_index_list[x]:
                # if there is sequence between the current and previous motifs, that sequence is a flank
                # add the flank's start and end coordinates to the emtpy list
                flank_pos_list.append(motif_index_list[x-1]+mot_len)
                flank_pos_list.append(motif_index_list[x]-1)
    return flank_pos_list

def all_substrings(string, min_length=2, max_length=None):
    # function that returns all possible substrings of the provided string
    # within the given size length limiits
    if max_length == None:
        max_length = len(string)
        substring_list = [string[start:start+length] for length in range(min_length, max_length+1) for start in range(len(string)-length+1)]
        substring_list.append(string)
    else:
        substring_list = [string[start:start+length] for length in range(min_length, max_length+1) for start in range(len(string)-length+1)]
    return substring_list

def best_a_dna_seq(dna_seq, min_seq_length=2, max_seq_length=None):
    # function that takes the given the dna_sequence then picks out the subsequence
    # that is at least the min_seq_length and has the highest A-DNA probability
    if max_seq_length == None:
        max_seq_length = len(dna_seq)
    seqs_list = all_substrings(dna_seq, min_seq_length, max_seq_length)
    unique_seqs = list(Counter(seqs_list).keys())
    seqs_data = list(map(seq_to_data, unique_seqs))
    seqs_data_df = pd.DataFrame(data = seqs_data)
    a_dna_probs = AB_form_predictor(seqs_data_df)
    a_dna_probs_dict = {unique_seqs[i]:a_dna_probs[i] for i in range(len(unique_seqs))}
    max_a_dna_prob = max(a_dna_probs)
    max_a_dna_seqs = dict(filter(lambda a_prob: a_prob[1]==max_a_dna_prob, a_dna_probs_dict.items()))
    return max_a_dna_seqs

def motif_promoters(promoter_df, motif, genome_dict):
    '''
    Function that indicates whether the set of supplied promoters contain the supplied motif
    and if so, how many time that motif appears in that promoter. The DataFrame returned
    does not filter out those promoters that do not contain the motif. Additionally, the 
    function checks both the forward and reverse complement, not just one orientation.

    Parameters
    ----------
    promoter_df : pandas DataFrame
        DataFrame containing the chromsome, TSS, promoter end coordinate, promoter length, and
        normalized signal of the promoters of interest.
    motif : string
        DNA sequence to scan for in motifs.
    genome_dict : dict
        Dictionary relaing each chromsome (key) to it sequence (value). This is used in
        conjunction wiht the promoter information to extract the promoter sequences.

    Returns
    -------
    motif_promoters_df : pandas DataFrame
        DataFrame containing the chromosome, strand, left and right promoter coordinates, and 
        the number of times the motif appears at the promoter. If a promoter loctaion does not
        contain the motif, it lists its strand as N/A
    '''
    promoter_motif_info = []
    for i in range(promoter_df.shape[0]):
        promoter_info = promoter_df.iloc[i].values.tolist()
        chromosome = promoter_info[1]
        promoter_lc = min(promoter_info[2], promoter_info[3])
        promoter_rc = max(promoter_info[2], promoter_info[3])
        fwd_promoter_seq = genome_dict[chromosome][promoter_lc: promoter_rc+1]
        rev_promoter_seq = seq_reverser(fwd_promoter_seq)
        if motif in fwd_promoter_seq:
            true_promoter_seq = fwd_promoter_seq
            motif_count = true_promoter_seq.count(motif)
            promoter_motif_info.append([chromosome, '+', promoter_lc, promoter_rc, motif_count])
        elif motif in rev_promoter_seq:
            true_promoter_seq = rev_promoter_seq
            motif_count = true_promoter_seq.count(motif) 
            promoter_motif_info.append([chromosome, '-', promoter_lc, promoter_rc, motif_count])
        else:
            promoter_motif_info.append([chromosome, 'N/A', promoter_lc, promoter_rc, 0])
    motif_promoters_df = pd.DataFrame(data=promoter_motif_info, columns = ['Chromosome', 'Strand', 'Promoter Left Coordinate', 'Promoter Right Coordinate', 'Motif Occurrences'])
    return motif_promoters_df

def flank_info(promoter_df, genome_dict, motif, reverse_complement = True):
    '''
    Function that finds promoters, wihtin the set of promoters provided, containing the 
    supplied motif. If a promoter sequence or its reverse complement contains the motif, the
    coordinates of the sequences flanking each motif occurence  as well a the length of each
    flank are returned in a pandas DataFrame.

    Parameters
    ----------
    promoter_df : pandas DataFrame
        DataFrame containing the chromsome, TSS, promter end coordinate, promoter length, and 
        normalized score for the promoters of interest.
    genome_dict : dict
        Dictionary relating each chromosome (key) in the genome to its sequence (value).
    motif : str
        Motif to search for represented as a string.
    reverse_complement : boolean
        If true, when searching for input motif, will also check for occurences of its reverse 
        complement. If False, will only check for the input motif. True by default.

    Returns
    -------
    promoter_flank_df : pandas DataFrame
        DataFrame containing the chromosome, strand, left and right promoter coordinates,
        left and right flank coordinates (relative to the chromsome like the promoter) and
        flank length.

    '''
    # create empty list to hold flank info that will be passed to DataFrame later
    promoter_flank_info = []
    # create a counter to keep track of the number of target promoters without a motif
    no_motif_counter = 0
    # determine if the reverse complement of input motif is to be checked along side the input motif
    if reverse_complement == True:
        # if checking for reverse complement motif as well, grab it from the input motif
        rev_motif = seq_reverser(motif)
        # for each promoter, grab its chromsome, coordinates, and score
        for i in range(promoter_df.shape[0]):
            promoter_info = promoter_df.iloc[i].values.tolist()
            chromosome = promoter_info[0]
            promoter_lc = min(promoter_info[1], promoter_info[2])
            promoter_rc = max(promoter_info[1], promoter_info[2])
            promoter_score = promoter_info[4]
            # use promoter coordinates, chromosome, and genome dictionary to get promoter sequence
            # then check it input and reverse complement motifs are in the sequence
            fwd_promoter_seq = genome_dict[chromosome][promoter_lc: promoter_rc+1]
            if (motif not in fwd_promoter_seq) and (rev_motif not in fwd_promoter_seq):
                # if neither motif is found, add 1 to no motif counter
                no_motif_counter += 1
            else:
                # if promoter containts at least 1 motif occurence, get flanking sequence and info
                # only need one strand since checking for forward and reverse complment of motif
                true_promoter_seq = fwd_promoter_seq
                # grab coordinates for each motif occurence as well as the sequences for the flanks
                motif_positions, flank_list = flank_finder(true_promoter_seq, motif, reverse_complement = reverse_complement)
                # grab the positions of each flank
                flank_positions = flank_position_finder(true_promoter_seq, motif_positions, motif)
                # for each flank, want its sequence, coordinates, chromosome, promoter coordinates, promoter score
                for flank_pos_in in np.arange(0,len(flank_positions),2):
                    flank_lc = flank_positions[flank_pos_in] + promoter_lc
                    flank_rc = flank_positions[flank_pos_in+1] + promoter_lc
                    flank_seq = genome_dict[chromosome][flank_lc: flank_rc + 1]
                    flank_length = flank_rc - flank_lc + 1
                    # append flank info to previously created empty list
                    promoter_flank_info.append([chromosome, promoter_lc, promoter_rc, promoter_score, flank_lc, flank_rc, flank_length, flank_seq])
        # create a DataFrame for flank information
        promoter_flank_df = pd.DataFrame(data = promoter_flank_info, columns = ['Chromosome', 'Left Promoter Coordinate', 'Right Promoter Coordinate', 'Normalized Promoter Signal', 'Left Flank Coordinate', 'Right Flank Coordinate', 'Flank Length', 'Sequence'])
    else:
        # do the same as above, but only searching for occurences of input motif
        # only difference is that the sequence strand is included
        for i in range(promoter_df.shape[0]):
            promoter_info = promoter_df.iloc[i].values.tolist()
            chromosome = promoter_info[0]
            promoter_lc = min(promoter_info[1], promoter_info[2])
            promoter_rc = max(promoter_info[1], promoter_info[2])
            promoter_score = promoter_info[4]
            fwd_promoter_seq = genome_dict[chromosome][promoter_lc: promoter_rc+1]
            rev_promoter_seq = seq_reverser(fwd_promoter_seq)
            if (motif not in fwd_promoter_seq) and (motif not in rev_promoter_seq):
                no_motif_counter += 1
            else:
                if motif in fwd_promoter_seq:
                    true_promoter_seq = fwd_promoter_seq
                    strand = '+'
                elif motif in rev_promoter_seq:
                    true_promoter_seq = rev_promoter_seq
                    strand = '-'
                motif_positions, flank_list = flank_finder(true_promoter_seq, motif, reverse_complement = reverse_complement)
                flank_positions = flank_position_finder(true_promoter_seq, motif_positions, motif)
                for flank_pos_in in np.arange(0,len(flank_positions),2):
                    flank_lc = flank_positions[flank_pos_in] + promoter_lc
                    flank_rc = flank_positions[flank_pos_in+1] + promoter_lc
                    flank_seq = genome_dict[chromosome][flank_lc: flank_rc + 1]
                    flank_length = flank_rc - flank_lc + 1
                    promoter_flank_info.append([chromosome,strand, promoter_lc, promoter_rc, promoter_score, flank_lc, flank_rc, flank_length, flank_seq])
            promoter_flank_df = pd.DataFrame(data = promoter_flank_info, columns = ['Chromosome', 'Strand', 'Left Promoter Coordinate', 'Right Promoter Coordinate', 'Normalized Promoter Signal', 'Left Flank Coordinate', 'Right Flank Coordinate', 'Flank Length', 'Sequence'])
            # filter out any flanks that are double counted because of overlap between promoters on different strands
    promoter_unique_flank_df = promoter_flank_df.drop_duplicates(subset=['Chromosome', 'Left Flank Coordinate', 'Right Flank Coordinate'])
    print(f"Out of {promoter_df.shape[0]} promoters, {promoter_df.shape[0] - no_motif_counter} contained the motif")
    return promoter_unique_flank_df


