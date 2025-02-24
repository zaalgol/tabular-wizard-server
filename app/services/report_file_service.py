import os
import math
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    XPreformatted
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from fastapi import HTTPException
from fastapi.responses import FileResponse
from werkzeug.utils import safe_join
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
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
            
    async def generate_model_details_file(self, model, df):
        SAVED_MODEL_FOLDER = os.path.join(
            self.config.SAVED_MODELS_FOLDER, model.user_id, model.model_name
        )
        details_filename = f"{model.model_name}__details.pdf"
        details_filepath = os.path.join(SAVED_MODEL_FOLDER, details_filename)
        
        if not os.path.exists(SAVED_MODEL_FOLDER):
            os.makedirs(SAVED_MODEL_FOLDER)
            
        doc = SimpleDocTemplate(details_filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name='Title',
            parent=styles['Title'],
            textColor='blue',
            alignment=TA_CENTER
        )
        
        flowables = []
        
        # Numeric columns for plotting
        numeric_cols = df.select_dtypes(include=['number'])

        # 1) Heatmap
        await self.__append_heatmap(flowables, model, title_style, numeric_cols, SAVED_MODEL_FOLDER)

        # 2) Scatter & Density
        await self.__append_scatter_and_density_plots(flowables, model, title_style, numeric_cols, SAVED_MODEL_FOLDER)

        # 3) Model Details
        await self.__append_model_details(flowables, model, styles, title_style)  

        # 4) Evaluations
        await self.__append_evaluations(flowables, model, title_style)

        # Build PDF
        doc.build(flowables)

        server_name = self.config.SERVER_NAME
        return f"http://{server_name}/download/{details_filename}"


    async def __append_heatmap(self, flowables, model, title_style, numeric_cols, SAVED_MODEL_FOLDER):
        """
        Heatmap using your existing approach. 
        Heatmaps are 'axes-level' in Seaborn, so they do work with plt.figure().
        """
        heatmap_filepath = os.path.join(
            SAVED_MODEL_FOLDER, f"{model.model_name}_heatmap.png"
        )

        # We create a figure with plt.figure(...),
        # then call sns.heatmap(...) on the current axes, so this is fine:
        fig = plt.figure(figsize=(10,8), dpi=300)
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.tight_layout()
        fig.savefig(heatmap_filepath, format='png')
        plt.close(fig)

        flowables.append(Paragraph("Heatmap", title_style))
        flowables.append(Spacer(1, 12))
        heatmap_image = Image(heatmap_filepath, width=6*inch, height=4.8*inch)
        flowables.append(heatmap_image)
        flowables.append(Spacer(1, 12))


    async def __append_scatter_and_density_plots(
        self, flowables, model, title_style, numeric_cols, SAVED_MODEL_FOLDER
    ):
        if numeric_cols.shape[1] == 0:
            return

        # ---- Pairplot ----
        pairplot_filepath = os.path.join(
            SAVED_MODEL_FOLDER, f"{model.model_name}_pairplot.png"
        )
        g = sns.pairplot(numeric_cols)
        g.fig.set_size_inches(10, 10)
        g.savefig(pairplot_filepath, dpi=300)
        plt.close(g.fig)

        flowables.append(PageBreak())
        flowables.append(Paragraph("Pairplot (Scatter Matrix)", title_style))
        flowables.append(Spacer(1, 12))

        pairplot_image = Image(pairplot_filepath, width=6*inch, height=6*inch)
        flowables.append(pairplot_image)
        flowables.append(Spacer(1, 12))

        # ---- Density Plots in a grid ----
        density_filepath = os.path.join(
            SAVED_MODEL_FOLDER, f"{model.model_name}_density.png"
        )
        fig = self.__create_density_plots(numeric_cols)  # see function below
        fig.savefig(density_filepath, dpi=300)
        plt.close(fig)

        flowables.append(PageBreak())
        flowables.append(Paragraph("Density Plots", title_style))
        flowables.append(Spacer(1, 12))

        # If you know how many rows were used, you can scale the PDF image:
        # For simplicity, just pick something bigger than the single-row approach:
        # E.g., 7 inches wide, 9 inches tall:
        density_image = Image(density_filepath, width=6*inch, height=6*inch)
        flowables.append(density_image)
        flowables.append(Spacer(1, 12))


    def __create_density_plots(self, numeric_cols):
        import math
        columns = numeric_cols.columns
        n_cols = len(columns)
        ncols = 2
        nrows = math.ceil(n_cols / ncols)

        # Figure ~12 inches wide, 4 inches high per row
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows))
        axes = axes.ravel()  # flatten the array of Axes

        for idx, col in enumerate(columns):
            sns.histplot(numeric_cols[col], kde=True, ax=axes[idx])
            axes[idx].set_title(f"Density Plot for {col}")

        # Hide any unused subplots if n_cols is odd
        for extra_idx in range(idx+1, nrows*ncols):
            axes[extra_idx].set_visible(False)

        plt.tight_layout()
        return fig


    async def __append_model_details(self, flowables, model, styles, title_style):
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
    
    async def __append_evaluations(self, flowables, model, title_style):
        flowables.append(PageBreak())
        flowables.append(Paragraph("Evaluations", title_style))
        flowables.append(Spacer(1, 12))
        preformatted_style = ParagraphStyle(
            name='Preformatted',
            fontName='Courier',
            wordWrap='LTR',
            fontSize=12,
            leading=14
        )
        
        flowables.append(XPreformatted(model.formated_evaluations, preformatted_style))


    async def download_file(self, user_id, model_name, filename, saved_folder):
        try:
            file_directory = safe_join(saved_folder, user_id, model_name)
            file_path = safe_join(os.getcwd(), file_directory, filename)
            
            if not os.path.isfile(file_path):
                raise HTTPException(status_code=404, detail="File not found")

            return FileResponse(file_path, filename=filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
