from pipelines.tables_reconstruction_pipeline import run_tables_reconstruction_pipeline


testing_file = r"C:\github\table_lense\pipeline_data\output\testing_file.csv"
output_dir = r"C:\github\table_lense\pipeline_data\output"


run_tables_reconstruction_pipeline(inference_csv=testing_file, output_dir=output_dir)
