test_files=("test_ops.py")
args="PYTHONPATH=. DEBUG=TRUE"

# run all
for file in ${test_files[@]}; do
  eval $args python3 tests/$file
done
