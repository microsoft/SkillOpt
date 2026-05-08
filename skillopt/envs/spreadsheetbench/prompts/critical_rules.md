## Critical Rules (MUST follow)
1. NEVER write Excel formulas to cells that will be graded on their displayed value.
   openpyxl does NOT compute formulas -- the evaluator will see None.
   Instead, compute results in Python and write literal values (numbers/strings).
2. After saving the workbook, ALWAYS reopen and verify the written values:
   `wb2 = openpyxl.load_workbook(OUTPUT_PATH); print(wb2[sheet][cell].value)`
3. Use the `write_file` tool to create solution.py -- it avoids shell escaping issues.
   Do NOT use `echo "..." > solution.py` for multi-line scripts.

