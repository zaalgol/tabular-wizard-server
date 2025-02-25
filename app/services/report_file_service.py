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
            
    async def generate_model_details_file(self, model, df, target_column_classes=None):
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
        await self.__append_scatter_and_density_plots(flowables, model, title_style, numeric_cols, SAVED_MODEL_FOLDER, len(df))

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
        Create and save a heatmap of numeric columns.
        """
        if numeric_cols.empty:
            return

        heatmap_filepath = os.path.join(
            SAVED_MODEL_FOLDER, f"{model.model_name}_heatmap.png"
        )

        fig = plt.figure(figsize=(10, 8), dpi=300)
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
        self, flowables, model, title_style, numeric_cols, SAVED_MODEL_FOLDER, row_count
    ):
        """
        Create and insert a pairplot (scatter matrix) and density plots in a grid.
        The point size (s) and alpha are chosen based on the number of rows.
        """
        if numeric_cols.shape[1] == 0:
            return

        # Decide dot parameters based on row_count
        dot_params = self.__compute_dot_params(row_count)

        # --- Pairplot (Scatter Matrix) ---
        pairplot_filepath = os.path.join(
            SAVED_MODEL_FOLDER, f"{model.model_name}_pairplot.png"
        )

        g = sns.pairplot(
            numeric_cols,
            diag_kind='hist',
            plot_kws=dot_params
        )
        g.fig.set_size_inches(10, 10)

        # Center the axis labels
        for row_axes in g.axes:
            for ax in row_axes:
                if ax is not None:
                    # x=0.5 => horizontal center, y=-0.15 => below x-axis
                    ax.xaxis.set_label_coords(0.5, -0.3)
                    # y=0.5 => vertical center, x=-0.15 => left of y-axis
                    ax.yaxis.set_label_coords(-0.3, 0.5)

        g.fig.tight_layout()
        g.savefig(pairplot_filepath, dpi=300)
        plt.close(g.fig)

        flowables.append(PageBreak())
        flowables.append(Paragraph("Pairplot (Scatter Matrix)", title_style))
        flowables.append(Spacer(1, 12))

        pairplot_image = Image(pairplot_filepath, width=6*inch, height=6*inch)
        flowables.append(pairplot_image)
        flowables.append(Spacer(1, 12))

        # --- Density Plots in a grid ---
        density_filepath = os.path.join(
            SAVED_MODEL_FOLDER, f"{model.model_name}_density.png"
        )
        fig = self.__create_density_plots(numeric_cols)
        fig.savefig(density_filepath, dpi=300)
        plt.close(fig)

        flowables.append(PageBreak())
        flowables.append(Paragraph("Density Plots", title_style))
        flowables.append(Spacer(1, 12))

        density_image = Image(density_filepath, width=6*inch, height=6*inch)
        flowables.append(density_image)
        flowables.append(Spacer(1, 12))


    def __compute_dot_params(self, row_count):
        """
        Return a dict of plot_kws with s (size) and alpha (transparency) 
        that scale according to the number of data points.
        """
        if row_count < 300:
            # Small dataset => bigger dots, less transparent
            return {'s': 20, 'alpha': 0.8, 'edgecolor': 'none'}
        elif row_count < 3000:
            # Medium => moderate size/transparency
            return {'s': 8, 'alpha': 0.5, 'edgecolor': 'none'}
        else:
            # Large => very small, more transparent
            return {'s': 2, 'alpha': 0.3, 'edgecolor': 'none'}


    def __create_density_plots(self, numeric_cols):
        """
        Create a grid of hist+kde (density) plots for the numeric columns.
        Arranged in 2 columns with as many rows as needed.
        """
        columns = numeric_cols.columns
        n_cols = len(columns)
        if n_cols == 0:
            return plt.figure()  # empty figure

        ncols = 2
        nrows = math.ceil(n_cols / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4*nrows))
        axes = axes.ravel()  # Flatten to iterate easily

        for idx, col in enumerate(columns):
            sns.histplot(numeric_cols[col], kde=True, ax=axes[idx])
            axes[idx].set_title(f"Density Plot for {col}")

        # Hide any subplots not used (e.g. if n_cols is odd)
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
