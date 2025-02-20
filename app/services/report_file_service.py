import os
import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, XPreformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from fastapi import HTTPException
from fastapi.responses import FileResponse
from werkzeug.utils import safe_join
import seaborn as sns
import matplotlib.pyplot as plt
from app.config.config import Config 

class ReportFileService:
    _instance = None

    def __init__(self) -> None:
        self.config = Config

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
            
    def generate_model_evaluations_file(self, model, df):
        SAVED_MODEL_FOLDER = os.path.join(self.config.SAVED_MODELS_FOLDER, model.user_id, model.model_name)
        evaluations_filename = f"{model.model_name}__evaluations.pdf"
        evaluations_filepath = os.path.join(SAVED_MODEL_FOLDER, evaluations_filename)
        
        if not os.path.exists(SAVED_MODEL_FOLDER):
            os.makedirs(SAVED_MODEL_FOLDER)
            
        doc = SimpleDocTemplate(evaluations_filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(name='Title', parent=styles['Title'], textColor='blue', alignment=TA_CENTER)
        preformatted_style = ParagraphStyle(name='Preformatted', fontName='Courier', wordWrap='LTR', fontSize=12, leading=14)
        
        flowables = []
        
        numeric_cols = df.select_dtypes(include=['number'])
        
        heatmap_filepath = os.path.join(SAVED_MODEL_FOLDER, f"{model.model_name}_heatmap.png")
        print("*" * 500 + f" SAVED_MODEL_FOLDER:{SAVED_MODEL_FOLDER},  evaluations_filepath:{evaluations_filepath}, heatmap_filepath:{heatmap_filepath}")
        self.__save_plot_as_image(lambda: sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f'),
                        heatmap_filepath, width=10, height=8, dpi=300)
    
        describe_df = numeric_cols.describe().transpose()
        describe_df = describe_df.apply(pd.to_numeric, errors='coerce')
        describe_heatmap_filepath = os.path.join(SAVED_MODEL_FOLDER, f"{model.model_name}_describe_heatmap.png")
        self.__save_plot_as_image(lambda: sns.heatmap(describe_df, annot=True, cmap='viridis', fmt='.2f'),
                        describe_heatmap_filepath, width=15, height=12, dpi=300)

        flowables.append(Paragraph("Heatmap", title_style))
        flowables.append(Spacer(1, 12))

        heatmap_image = Image(heatmap_filepath, width=6*inch, height=4.8*inch)
        flowables.append(heatmap_image)
        flowables.append(Spacer(1, 12))

        flowables.append(PageBreak())
        flowables.append(Paragraph("Describe Heatmap", title_style))
        flowables.append(Spacer(1, 12))

        describe_heatmap_image = Image(describe_heatmap_filepath, width=6*inch, height=4.8*inch)
        flowables.append(describe_heatmap_image)
        flowables.append(Spacer(1, 12))

        flowables.append(PageBreak())
        flowables.append(Paragraph("Model Details", title_style))
        flowables.append(Spacer(1, 12))

        text = (
            f"Model Name: {model.model_name}\n"
            f"Model Type: {model.model_type}\n"
            f"Training Strategy: {model.training_strategy}\n"
            f"Sampling Strategy: {model.sampling_strategy}\n"
            f"Metric: {model.metric}\n\n"
        )
        for line in text.split('\n'):
            flowables.append(Paragraph(line, styles['Normal']))
            flowables.append(Spacer(1, 12))
            
        flowables.append(PageBreak())
        flowables.append(Paragraph("Evaluations", title_style))
        flowables.append(Spacer(1, 12))
        
        flowables.append(XPreformatted(model.formated_evaluations, preformatted_style))

        doc.build(flowables)

        # scheme = 'https' if self.config.PREFERRED_URL_SCHEME == 'https' else 'http'
        server_name = self.config.SERVER_NAME
        return f"http://{server_name}/download/{evaluations_filename}"
    
    def __save_plot_as_image(self, plot_func, filepath, width, height, dpi=300):
        try:
            fig = plt.figure(figsize=(width, height), dpi=dpi)
            plot_func()
            plt.gca().set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha='right')
            plt.gca().set_yticklabels(plt.gca().get_yticklabels(), rotation=0)
            
            # Set format for the annotations
            for text in plt.gca().texts:
                text.set_text(f'{float(text.get_text()):.2f}')
            
            plt.tight_layout()
            fig.savefig(filepath, format='png')
            plt.close(fig)
            print(f"Saved plot to {filepath}")
        except Exception as e:
            print(f"Error saving plot as image: {e}")

    async def download_file(self, user_id, model_name, filename, saved_folder):
        try:
            # if file_type == 'inference':
            #     saved_folder = Config.SAVED_INFERENCES_FOLDER
            # else: 
            #     saved_folder = Config.SAVED_MODELS_FOLDER
            file_directory = safe_join(saved_folder, user_id, model_name)
            file_path = safe_join(os.getcwd(), file_directory, filename)
            
            if not os.path.isfile(file_path):
                raise HTTPException(status_code=404, detail="File not found")

            return FileResponse(file_path, filename=filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
