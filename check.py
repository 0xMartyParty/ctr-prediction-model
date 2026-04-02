#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:41:22 2025

@author: maxmarte
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 ONLY: Check that ProjectTestData.csv and ProjectSubmission-TeamX.csv
have the same number of rows AND the same row order (by comparing full 'id' sequence).

Run:
  python3 check_test_vs_submission.py
"""

import os
import sys
import numpy as np
import pandas as pd


# =========================
# CONFIG (EDIT THESE)
# =========================
BASE = "/Users/maxmarte/Desktop/AI final Project"
TEAM_NUM = 8  # <-- CHANGE to your team number (e.g., 10)

PROJECT_TEST = os.path.join(BASE, "ProjectTestData.csv")
SUBMISSION_TEMPLATE = os.path.join(BASE, f"ProjectSubmission-Team{TEAM_NUM}.csv")


def stop(msg: str, code: int = 1):
    print("\n" + "=" * 70)
    print("STOP:", msg)
    print("=" * 70)
    sys.exit(code)


def main():
    print("Loading files...")
    test_df = pd.read_csv(PROJECT_TEST, usecols=["id"])
    sub_df  = pd.read_csv(SUBMISSION_TEMPLATE, usecols=["id", "P(click)"])

    n_test = len(test_df)
    n_sub  = len(sub_df)

    print(f"ProjectTestData.csv rows:         {n_test:,}")
    print(f"ProjectSubmission-TeamX.csv rows: {n_sub:,}")

    if n_test != n_sub:
        stop(
            "Row counts do NOT match. Because 'id' is not guaranteed unique, "
            "do NOT merge/reorder by id. You need matching files."
        )

    test_ids = test_df["id"].astype(str).to_numpy()
    sub_ids  = sub_df["id"].astype(str).to_numpy()

    if np.array_equal(test_ids, sub_ids):
        print("✓ PASS: Row counts match and id order matches exactly.")
        return

    # Find and report first mismatch
    mismatch_idx = np.where(test_ids != sub_ids)[0]
    first = int(mismatch_idx[0])

    print("\n✗ FAIL: id order does NOT match.")
    print(f"First mismatch at row index: {first}")
    print(f"  ProjectTestData id:        {test_ids[first]}")
    print(f"  Submission template id:    {sub_ids[first]}")
    print("\nBecause 'id' is not unique, aligning by id is unsafe. You need a correctly ordered template.")


if __name__ == "__main__":
    main()
