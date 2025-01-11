from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']

class AnalysisForm(forms.Form):
    ANALYSIS_CHOICES = [
        ('univariate', 'Analyse Univariée'),
        ('bivariate', 'Analyse Bivariée')
    ]
    
    analysis_type = forms.ChoiceField(choices=ANALYSIS_CHOICES)
    variable_1 = forms.ChoiceField(required=True)
    variable_2 = forms.ChoiceField(required=False)
    
    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns', None)
        super().__init__(*args, **kwargs)
        if columns:
            self.fields['variable_1'].choices = [(col, col) for col in columns]
            self.fields['variable_2'].choices = [('', '----')] + [(col, col) for col in columns]