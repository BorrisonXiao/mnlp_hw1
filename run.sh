#!/usr/bin/env bash
# Cihan Xiao 2022

set -eou pipefail

stage=0
stop_stage=100
python=python3
dumpdir=dump
expdir=exp
m1iter=30
m3iter=5
m4iter=20
m5iter=10
nj=72
translate=false

. ./scripts/parse_options.sh || exit 1

log() {
    # This function is from espnet
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
    local a b
    a=$1
    for b in "$@"; do
        if [ "${b}" -le "${a}" ]; then
            a="${b}"
        fi
    done
    echo "${a}"
}
SECONDS=0

source /home/cxiao7/miniconda3/etc/profile.d/conda.sh
conda activate mnlp

# Note: Deprecated, as GIZA++ has support for the vocab creation
# if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
#     mkdir -p ${dumpdir}
#     # Make the vocab files
#     log "Making the vocab files"
#     ${python} ./scripts/make_vocab.py \
#         --input ./data/bitext.en \
#         --output ./${dumpdir}/S.vcb

#     ${python} ./scripts/make_vocab.py \
#         --input ./data/bitext.sv \
#         --output ./${dumpdir}/T.vcb \
#         --lang "sv"
# fi

# if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
#     # Make the bitext file
#     log "Making the bitext file"
#     ${python} ./scripts/make_bitext.py \
#         --src ./data/bitext.en \
#         --tgt ./data/bitext.sv \
#         --output ./${dumpdir}/ST.snt \
#         --src-vocab ./${dumpdir}/S.vcb \
#         --tgt-vocab ./${dumpdir}/T.vcb \
#         --src-lang "en" \
#         --tgt-lang "sv"
# fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Preprocess the bitext files"
    ${python} ./scripts/normalize_bitext.py \
        --input data/bitext.en \
        --conllu-file ./data/en_lines-ud-train.conllu \
        --output ./${dumpdir}/bitext.en

    ${python} ./scripts/normalize_bitext.py \
        --input data/bitext.sv \
        --output ./${dumpdir}/bitext.sv

    ./giza-pp/GIZA++-v2/plain2snt.out \
        ./${dumpdir}/bitext.en \
        ./${dumpdir}/bitext.sv
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Compile the cooccurence file"
    ./giza-pp/GIZA++-v2/snt2cooc.out \
        ./${dumpdir}/bitext.en.vcb \
        ./${dumpdir}/bitext.sv.vcb \
        ./${dumpdir}/bitext.en_bitext.sv.snt \
        >./${dumpdir}/ST.cooc
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Run GIZA++"
    _expdir=${expdir}/$(date +"%Y-%m-%d-%H-%M-%S")
    mkdir -p "${_expdir}"
    ./giza-pp/GIZA++-v2/GIZA++ \
        -S ./${dumpdir}/bitext.en.vcb \
        -T ./${dumpdir}/bitext.sv.vcb \
        -C ./${dumpdir}/bitext.en_bitext.sv.snt \
        -CoocurrenceFile ./${dumpdir}/ST.cooc \
        -outputpath "./${_expdir}" \
        -t1 1 \
        -t345 1 \
        -model1iterations $m1iter \
        -model4iterations $m4iter \
        -model1dumpfrequency 5 \
        -model345dumpfrequency 5
    ln -snfv "$PWD/${_expdir}" "$PWD/${expdir}/latest"
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Project the conllu file"
    _expdir=${expdir}/latest
    ${python} ./scripts/project.py \
        --src ./data/en_lines-ud-train.conllu \
        --A1-file ./${_expdir}/*.A1.$m1iter \
        --output ./${expdir}/projected.A1.conllu

    # let finalm4iter=$m3iter+$m4iter
    ${python} ./scripts/project.py \
        --src ./data/en_lines-ud-train.conllu \
        --A1-file ./${_expdir}/*.A3.final \
        --output ./${expdir}/projected.A4.conllu
fi

if ${translate}; then
    if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
        log "Translate the additional English text into Swedish"

        key_file=data/dev_pos_bitext.en
        _logdir=${expdir}/logdir
        mkdir -p "${_logdir}"
        _nj=$(min "${nj}" "$(wc <${key_file} -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/text.${n}.scp"
        done
        # shellcheck disable=SC2086
        scripts/split_scp.pl "${key_file}" ${split_scps}

        scripts/queue.pl --mem 8G --gpu "0" JOB=1:"${_nj}" "${_logdir}"/m2m100_418M.JOB.log \
            ./scripts/translate.py \
            --model "facebook/m2m100_418M" \
            --src ${_logdir}/text.JOB.scp \
            --output ${_logdir}/output.m2m100_418M.JOB.sv ||
            {
                cat $(grep -l -i error "${_logdir}"/m2m100_418M.*.log)
                exit 1
            }

        for i in $(seq "${_nj}"); do
            if [ -f "${_logdir}/output.m2m100_418M.${i}.sv" ]; then
                cat "${_logdir}/output.m2m100_418M.${i}.sv"
            fi
        done >"data/dev_pos_bitext.sv"
    fi

    if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
        log "Translate the additional English text into Swedish using m2m100_1.2B"

        key_file=data/dev_pos_bitext.en
        _logdir=${expdir}/logdir
        mkdir -p "${_logdir}"
        _nj=$(min "${nj}" "$(wc <${key_file} -l)")
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/text.${n}.scp"
        done
        # shellcheck disable=SC2086
        scripts/split_scp.pl "${key_file}" ${split_scps}

        scripts/queue.pl --mem 8G --gpu "0" JOB=1:"${_nj}" "${_logdir}"/m2m100_1.2B.JOB.log \
            ./scripts/translate.py \
            --model "facebook/m2m100_1.2B" \
            --src ${_logdir}/text.JOB.scp \
            --output ${_logdir}/output.m2m100_1.2B.JOB.sv ||
            {
                cat $(grep -l -i error "${_logdir}"/m2m100_1.2B.*.log)
                exit 1
            }

        for i in $(seq "${_nj}"); do
            if [ -f "${_logdir}/output.m2m100_1.2B.${i}.sv" ]; then
                cat "${_logdir}/output.m2m100_1.2B.${i}.sv"
            fi
        done >"data/dev_pos_bitext.m2m100_1.2B.sv"
    fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Concatenate the original bitext and the additional bitext and normalize them"

    # First concatenate the original bitext and the additional bitext
    cat ./data/bitext.en ./data/dev_pos_bitext.en >data/concat.en
    cat ./data/bitext.sv ./data/dev_pos_bitext.sv >data/concat.sv
    cat ./data/bitext.sv ./data/dev_pos_bitext.m2m100_1.2B.sv >data/concat.m2m100_1.2B.sv

    # Then, normalize the new bitext and concatenate it with the original tokenized bitext
    ${python} ./scripts/normalize_bitext.py \
        --input data/dev_pos_bitext.en \
        --conllu-file ./data/en_lines-ud-dev.conllu \
        --output ${dumpdir}/dev_pos_bitext.en
    cat ${dumpdir}/bitext.en ${dumpdir}/dev_pos_bitext.en >${dumpdir}/concat.en

    ${python} ./scripts/normalize_bitext.py \
        --input data/dev_pos_bitext.sv \
        --output ${dumpdir}/dev_pos_bitext.sv
    cat ${dumpdir}/bitext.sv ${dumpdir}/dev_pos_bitext.sv >${dumpdir}/concat.sv

    ${python} ./scripts/normalize_bitext.py \
        --input data/concat.m2m100_1.2B.sv \
        --output ${dumpdir}/dev_pos_bitext.m2m100_1.2B.sv
    cat ${dumpdir}/bitext.sv ${dumpdir}/dev_pos_bitext.m2m100_1.2B.sv >${dumpdir}/concat_12B.sv

    ./giza-pp/GIZA++-v2/plain2snt.out \
        ./${dumpdir}/concat.en \
        ./${dumpdir}/concat.sv

    ./giza-pp/GIZA++-v2/plain2snt.out \
        ./${dumpdir}/concat.en \
        ./${dumpdir}/concat_12B.sv

    ./giza-pp/GIZA++-v2/snt2cooc.out \
        ./${dumpdir}/concat.en.vcb \
        ./${dumpdir}/concat.sv.vcb \
        ./${dumpdir}/concat.en_concat.sv.snt \
        >./${dumpdir}/concat_ST.cooc

    ./giza-pp/GIZA++-v2/snt2cooc.out \
        ./${dumpdir}/concat.en.vcb \
        ./${dumpdir}/concat.sv.vcb \
        ./${dumpdir}/concat.en_concat_12B.sv.snt \
        >./${dumpdir}/concat_12B_ST.cooc
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Similarly, train the GIZA++ model on the concatenated bitext"

    _expdir=${expdir}/dev-$(date +"%Y-%m-%d-%H-%M-%S")
    mkdir -p "${_expdir}"
    ./giza-pp/GIZA++-v2/GIZA++ \
        -S ./${dumpdir}/concat.en.vcb \
        -T ./${dumpdir}/concat.sv.vcb \
        -C ./${dumpdir}/concat.en_concat.sv.snt \
        -CoocurrenceFile ./${dumpdir}/concat_ST.cooc \
        -outputpath "./${_expdir}" \
        -t1 1 \
        -t345 1 \
        -model1iterations $m1iter \
        -model4iterations $m4iter \
        -model1dumpfrequency 5 \
        -model345dumpfrequency 5
    ln -snfv "$PWD/${_expdir}" "$PWD/${expdir}/dev-latest"
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Similarly, train the GIZA++ model on the concatenated bitext using the 12B model"

    _expdir=${expdir}/dev-12B-$(date +"%Y-%m-%d-%H-%M-%S")
    mkdir -p "${_expdir}"
    ./giza-pp/GIZA++-v2/GIZA++ \
        -S ./${dumpdir}/concat.en.vcb \
        -T ./${dumpdir}/concat_12B.sv.vcb \
        -C ./${dumpdir}/concat.en_concat_12B.sv.snt \
        -CoocurrenceFile ./${dumpdir}/concat_12B_ST.cooc \
        -outputpath "./${_expdir}" \
        -t1 1 \
        -t345 1 \
        -model1iterations $m1iter \
        -model4iterations $m4iter \
        -model5iterations $m5iter \
        -model1dumpfrequency 5 \
        -model345dumpfrequency 5
    ln -snfv "$PWD/${_expdir}" "$PWD/${expdir}/dev-12B-latest"
fi

if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
    log "Similarly, project the additional English text into Swedish using the new GIZA++ output"
    _expdir=${expdir}/dev-latest
    ${python} ./scripts/project.py \
        --src ./data/en_lines-ud-dev.conllu \
        --start -1032 \
        --A1-file ./${_expdir}/*.A3.final \
        --output ./${expdir}/projected.dev.A4.conllu
fi

if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
    log "Similarly, project the additional English text into Swedish using the new GIZA++ output"
    _expdir=${expdir}/dev-12B-latest
    ${python} ./scripts/project.py \
        --src ./data/en_lines-ud-dev.conllu \
        --start -1032 \
        --A1-file ./${_expdir}/*.A3.25 \
        --output ./${expdir}/projected.dev.12B.A4.conllu
    ${python} ./scripts/project.py \
        --src ./data/en_lines-ud-dev.conllu \
        --start -1032 \
        --A1-file ./${_expdir}/*.A3.final \
        --output ./${expdir}/projected.dev.12B.A5.conllu
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
