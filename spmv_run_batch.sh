folder=/media/matrices/ssget/MM/todas
list=lista9.txt

batched=0

rm out.txt
while read line; do
  matrix="$(basename -- ${line%})"

  echo "Working on $matrix"

  #compute-sanitizer ./bmsparse_float "$folder" "$matrix" "$matrix" "$batched" >> "out.txt"
  ./bmsparse_spmv_float "$folder" "$matrix" "$matrix" "$batched" >> "spmv_out.txt"

done < $list