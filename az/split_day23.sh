last_day=day_23
lines=`wc -l $last_day | awk '{print $1}'`
echo $lines
former=89137319
latter=89137318

head -n $former $last_day > day_23_test
tail -n $latter $last_day > day_23_validation
