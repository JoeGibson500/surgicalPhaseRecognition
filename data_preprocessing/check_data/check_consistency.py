from consistency_check import check_missing_files, check_column_consistency, extract_unique_phase_labels


# Run all consistency checks on the dataset
check_missing_files()
check_column_consistency()
extract_unique_phase_labels()

print("Consistency check completed. Results saved in reports/logs/data_consistency.log")
