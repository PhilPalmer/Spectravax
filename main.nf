#!/usr/bin/env nextflow

// Create channel for peptides txt file
// TODO: Generate the peptides file from the input fasta file
ch_peptides = Channel.value(file( "${params.peptides}" ))
    .splitText( by: params.chunk_size, file: true)
    .take( 3 ) // TODO: Remove this line

// Create channel for HLA alleles txt file
ch_hla_alleles = Channel
    .value(file( "${params.hla_alleles}" ))
    .splitText()
    .map { allele -> allele.trim() }
    .take( 3 ) // TODO: Remove this line

ch_allele_peptides = ch_hla_alleles
    .combine(ch_peptides)

// Run the NetMHCpan process for each chunk of peptides and each HLA allele
process NetMHCpan {
    tag "${allele_clean} ${peptides_idx}"

    input:
    set val(allele), file(peptides_chunk) from ch_allele_peptides
    
    output:
    set val(allele_clean), file(allele_peptides_preds) into ch_allele_peptides_preds

    script:
    peptides_idx = peptides_chunk.name.split("\\.")[-2]
    allele_clean = allele.replace("*", "_").replace(":", "")
    allele_peptides_preds = "${allele_clean}_p${peptides_idx}_preds.xls"
    peptides_flag = params.mhc1 ? "-p" : "-inptype 1 -f"
    """
    ${params.netmhcpan_cmd} ${peptides_flag} ${peptides_chunk} -a ${allele} -BA -xls -xlsfile ${allele_peptides_preds}
    """
  }

// Group the predictions by allele so that we can merge the predictions for each peptide chunk
ch_allele_group_preds = ch_allele_peptides_preds
    .groupTuple(by: 0)

process MergeNetMHCpan {
    tag "${allele}"
    publishDir "${params.outdir}/MHC_Binding/preds", mode: 'copy'

    input:
    set val(allele), file(allele_peptides_preds) from ch_allele_group_preds

    output:
    set val(allele), file(allele_preds) into ch_allele_preds

    script:
    allele_preds = "${allele}_preds.xls"
    """
    head -n 2 -q ${allele_peptides_preds[0]} > ${allele_preds}
    tail -n +3 -q ${allele_peptides_preds} >> ${allele_preds}
    """
}
