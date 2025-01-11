from django.db import models
import pandas as pd
import json

class Dataset(models.Model):
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns = models.JSONField(null=True)
    numeric_columns = models.JSONField(null=True)
    categorical_columns = models.JSONField(null=True)
    row_count = models.IntegerField(null=True)
    
    def process_file(self):
        df = pd.read_csv(self.file.path)
        self.columns = df.columns.tolist()
        self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        self.row_count = len(df)
        self.save()
        
    def get_dataframe(self):
        return pd.read_csv(self.file.path)
        
    def __str__(self):
        return self.name