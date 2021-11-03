#!/usr/bin/zsh
cd /home/jacob/Documents/learning/data-science-infinity/data
sheet_names=$(in2csv grocery_database.xlsx -n)
sheet_names=("${(f)sheet_names}")
for sheet in $sheet_names;
do
    if [$sheet = "database_info"]; then
        in2csv grocery_database.xlsx --sheet $sheet | tail -n+3 > $sheet.csv;
    elif [$sheet = "transactions"]; then
        in2csv grocery_database.xlsx --sheet $sheet | csvcut -C 7 > $sheet.csv;
    else;
        in2csv grocery_database.xlsx --sheet $sheet > $sheet.csv;
    fi
done