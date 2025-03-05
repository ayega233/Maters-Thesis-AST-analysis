import pyarrow.parquet as pq
import pandas as pd
import re

class FileProcessor:
    def __init__(self, parquet_file, batch_size=10000):
        self.parquet_file = parquet_file
        self.batch_size = batch_size
        self.table = None
    
    def load_parquet(self):        
        self.table = pq.read_table(self.parquet_file).to_pandas()        
    
    def clean_content(self):       
        if 'content' in self.table.columns:
            self.table['content'] = self.table['content'].apply(lambda x: re.sub(r'[^\x20-\x7E\n\r\t]', '', str(x)))
    
    def remove_timezone(self):       
        if 'datetime' in self.table.columns:
            self.table['datetime'] = self.table['datetime'].dt.tz_localize(None)
    
    def split_and_save(self):        
        if 'repo' in self.table.columns:
            unique_values = self.table['repo'].unique()
            for value in unique_values:
                filtered_df = self.table[self.table['repo'] == value]
                filename = f"filename_{value}.xlsx"
                filtered_df.to_excel(filename, index=False)
                print(f"Saved {filename}")
    
    def process(self):        
        self.load_parquet()
        self.clean_content()
        self.remove_timezone()
        self.split_and_save()
        print("Processing completed successfully!")

if __name__ == "__main__":
    processor = FileProcessor("defectors/line_bug_prediction_splits/random/train.parquet.gzip")
    processor.process()