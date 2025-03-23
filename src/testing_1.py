from pipelines.tables_reconstruction_pipeline import run_tables_reconstruction_pipeline


# test_file = r"C:\github\table_lense\pipeline_data\output\testing_file.csv"
test_file = r"C:\github\table_lense\pipeline_data\output\test_file_1.csv"
output_dir = r"C:\github\table_lense\pipeline_data\output"


run_tables_reconstruction_pipeline(inference_csv=test_file, output_dir=output_dir)
