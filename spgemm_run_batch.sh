folder=/media/matrices/ssget/MM/todas
list=lista9.txt

segmented=0
tc_version=5
verbose=0

rm spgemm_out.txt
while read line; do
  matrix="$(basename -- ${line%})"

  echo "Working on $matrix"

  #compute-sanitizer ./bmsparse_float "$folder" "$matrix" "$matrix" "$segmented" "$tc_version" "$verbose" >> "out.txt"
  ./bmsparse_spgemm_float "$folder" "$matrix" "$matrix" "$segmented" "$tc_version" "$verbose" >> "spgemm_out.txt"

done < $list