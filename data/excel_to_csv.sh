  458  sheet_names=$(in2csv grocery_database.xlsx -n)
  459  sheet_names=("${(f)sheet_names}")
  460  $sheet_names
  461  for sheet in $sheet_names; do in2csv grocery_database.xlsx --sheet $sheet;done
  462  for sheet in $sheet_names; do in2csv grocery_database.xlsx --sheet $sheet > $sheet.csv;done
  463  ls
  464  head campaign_data.csv
  465  head *.csv
  466  fc -l -5
  467  fc -l -10
  468  fc -l -9
  469  fc -l -11
