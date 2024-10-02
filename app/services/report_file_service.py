import os
import app.app as app
from flask import current_app

import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, XPreformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

import matplotlib.pyplot as plt

class ReportFileService:
    _instance = None

    def __init__(self) -> None:
        pass

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
            
    def generate_model_evaluations_file(self, model, dataset):
        # Emit an event for training success
        SAVED_MODEL_FOLDER = os.path.join(app.Config.SAVED_MODELS_FOLDER, model.user_id, model.model_name)
        evaluations_filename = f"{model.model_name}__evaluations.pdf"
        evaluations_filepath = os.path.join(SAVED_MODEL_FOLDER, evaluations_filename)
        
        if not os.path.exists(SAVED_MODEL_FOLDER):
            os.makedirs(SAVED_MODEL_FOLDER)
            
        # Create the PDF document
        doc = SimpleDocTemplate(evaluations_filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Define a blue title style
        title_style = ParagraphStyle(name='Title', parent=styles['Title'], textColor='blue', alignment=TA_CENTER)
        
        # Define a preformatted text style with larger font size and fixed-width font
        preformatted_style = ParagraphStyle(name='Preformatted', fontName='Courier', wordWrap='LTR', fontSize=12, leading=14)
        
        flowables = []

        # Select only numeric columns for correlation heatmap
        numeric_cols = dataset.select_dtypes(include=['number'])
        
        # Generate heatmap
        heatmap_filepath = os.path.join(SAVED_MODEL_FOLDER, f"{model.model_name}_heatmap.png")
        self.__save_plot_as_image(lambda: sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f'),
                        heatmap_filepath, width=10, height=8, dpi=300)

        # Generate transposed describe heatmap
        describe_df = dataset.describe().transpose()
        describe_heatmap_filepath = os.path.join(SAVED_MODEL_FOLDER, f"{model.model_name}_describe_heatmap.png")
        self.__save_plot_as_image(lambda: sns.heatmap(describe_df, annot=True, cmap='viridis', fmt='.2f'),
                        describe_heatmap_filepath, width=15, height=12, dpi=300)

        # Add "Heatmap" title
        flowables.append(Paragraph("Heatmap", title_style))
        flowables.append(Spacer(1, 12))

        # Add heatmap to PDF
        heatmap_image = Image(heatmap_filepath, width=6*inch, height=4.8*inch)
        flowables.append(heatmap_image)
        flowables.append(Spacer(1, 12))

        # Add "Describe Heatmap" title
        flowables.append(PageBreak())
        flowables.append(Paragraph("Describe Heatmap", title_style))
        flowables.append(Spacer(1, 12))

        # Add describe heatmap to PDF
        describe_heatmap_image = Image(describe_heatmap_filepath, width=6*inch, height=4.8*inch)
        flowables.append(describe_heatmap_image)
        flowables.append(Spacer(1, 12))

        # Add "Model Details" title
        flowables.append(PageBreak())
        flowables.append(Paragraph("Model Details", title_style))
        flowables.append(Spacer(1, 12))

        # Add text evaluations
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
            
        # Add "Evaluations" title
        flowables.append(PageBreak())
        flowables.append(Paragraph("Evaluations", title_style))
        flowables.append(Spacer(1, 12))
        
        # Add formatted evaluations to PDF using XPreformatted to preserve spacing and wrap text
        flowables.append(XPreformatted(model.formated_evaluations, preformatted_style))

        doc.build(flowables)

        # Generate a unique URL for the pdf file
        scheme = 'https' if current_app.config.get('PREFERRED_URL_SCHEME', 'http') == 'https' else 'http'
        server_name = current_app.config.get('SERVER_NAME', 'localhost:8080')
        return f"{scheme}://{server_name}/download/{evaluations_filename}"
    
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
