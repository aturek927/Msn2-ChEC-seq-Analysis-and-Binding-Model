#!/bash/bin

ref_gen=R64-1-1_index_files/R64-1-1

echo "Please enter the name of the Msn2 construct to be aligned (Full, DBD, or non_DBD)"

read construct

new_folder='Msn2_'$construct'_60sec_alignment_files_R64'
forward_read_file='Msn2_'$construct'_60sec_1.fastq'
reverse_read_file='Msn2_'$construct'_60sec_2.fastq'
output_file_name='Msn2_'$construct'_60sec_alignment.sam'
log_file='Msn2_'$construct'_60sec_alignment_log.log'
unsorted_bam_file='Msn2_'$construct'_60sec_alignment_bam.bam'
coord_sorted_bam_file='Msn2_'$construct'_60sec_alignment_coord_sorted_bam.bam'
name_sorted_bam_file='Msn2_'$construct'_60sec_alignment_name_sorted_bam.bam'
coord_sorted_bed_file='Msn2_'$construct'_60sec_alignment_coord_sorted_bed.bed'
name_sorted_bedpe_file='Msn2_'$construct'_60sec_alignment_name_sorted_bedpe.bedpe'
genome_coverage_file='Msn2_'$construct'_60sec_alignment_5prime_genome_coverage.counts'

if [ ! -d $new_folder ]; then
	echo $new_folder' does not exist, so it will be created'
	mkdir $new_folder
fi 

bowtie -m 1 --best -x $ref_gen -1 $forward_read_file -2 $reverse_read_file -S $new_folder'/'$output_file_name 2> $new_folder'/'$log_file

echo "The alignment has been completed"

head $new_folder'/'$log_file

cd $new_folder

samtools sort $output_file_name -o $coord_sorted_bam_file

samtools sort -n $output_file_name -o $name_sorted_bam_file

bedtools bamtobed -i $coord_sorted_bam_file > $coord_sorted_bed_file

bedtools bamtobed -bedpe -i $name_sorted_bam_file > $name_sorted_bedpe_file

bedtools genomecov -d -5 -fs 1 -ibam $coord_sorted_bam_file > $genome_coverage_file


