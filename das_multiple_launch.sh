for pos in 1 2 3 4 5 6 7 8
do
    nlprun -a pyvene -g 1 -r 40G -q sphinx -o "logs/das_multiple_$pos.log" 'bash das_multiple.sh '"$pos"
done