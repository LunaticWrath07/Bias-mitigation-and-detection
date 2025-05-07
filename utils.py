import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from io import BytesIO

def create_pdf_report(bias_metrics, mitigation_method, fairness_metrics):
    """
    Creates a PDF report with bias detection and mitigation metrics.
    
    Parameters:
    -----------
    bias_metrics : dict
        Dictionary containing bias detection metrics before mitigation
    mitigation_method : str
        The mitigation method selected by the user
    fairness_metrics : dict
        Dictionary containing fairness metrics after mitigation
    
    Returns:
    --------
    BytesIO buffer containing the PDF report
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    width, height = letter
    
    # Create list of flowables for the document
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Custom style for metrics
    metric_style = ParagraphStyle(
        'MetricStyle',
        parent=normal_style,
        leftIndent=20,
        spaceBefore=5,
        spaceAfter=5
    )
    
    # Title
    elements.append(Paragraph("Bias Detection and Mitigation Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Summary section
    elements.append(Paragraph("Executive Summary", heading_style))
    
    # Add summary text
    summary_text = """
    This report presents the results of bias detection and mitigation on the dataset. 
    It shows the bias metrics before mitigation and the fairness metrics after applying 
    the selected mitigation technique.
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    
    # Bias Detection Metrics
    elements.append(Paragraph("Bias Detection Metrics", heading_style))
    elements.append(Spacer(1, 10))
    
    # Create a table for bias metrics
    bias_data = [['Metric', 'Value', 'Ideal Range']]
    
    # Add metrics with ideal ranges
    ideal_ranges = {
        "Disparate Impact": "0.8 - 1.2",
        "Statistical Parity Difference": "-0.1 to 0.1",
        "Equal Opportunity Difference": "-0.1 to 0.1",
        "Average Odds Difference": "-0.1 to 0.1"
    }
    
    for metric, value in bias_metrics.items():
        bias_data.append([metric, f"{value:.2f}", ideal_ranges.get(metric, "N/A")])
    
    # Create the table
    bias_table = Table(bias_data, colWidths=[250, 100, 120])
    bias_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(bias_table)
    elements.append(Spacer(1, 20))
    
    # Mitigation Method
    elements.append(Paragraph("Mitigation Method", heading_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"The selected mitigation method was: <b>{mitigation_method}</b>", normal_style))
    
    # Add explanation of the mitigation method
    mitigation_explanations = {
        "Reweighting": """
        Reweighting adjusts the importance of training examples to ensure fairness. 
        It assigns higher weights to samples from disadvantaged groups and lower weights to samples from advantaged groups,
        helping to balance the impact of different demographic groups on the model.
        """,
        
        "Resampling": """
        Resampling modifies the dataset by oversampling minority groups and undersampling majority groups.
        This ensures balanced representation of all groups in the training data, which can help reduce bias
        in the model's predictions.
        """,
        
        "Fair Representation Learning": """
        Fair Representation Learning uses neural networks to transform the original features into a new representation
        that preserves the information needed for the main task while removing information about the sensitive attribute.
        This helps create features that are independent of protected attributes.
        """,
        
        "Adversarial Debiasing": """
        Adversarial Debiasing uses a two-part neural network: a predictor that tries to accurately predict the target variable,
        and an adversary that tries to predict the sensitive attribute from the predictor's internal state.
        The predictor learns to maximize task performance while minimizing the adversary's ability to determine the sensitive attribute.
        """
    }
    
    explanation = mitigation_explanations.get(mitigation_method, "No explanation available for this method.")
    elements.append(Paragraph(explanation, metric_style))
    elements.append(Spacer(1, 20))
    
    # Fairness Metrics After Mitigation
    elements.append(Paragraph("Fairness Metrics After Mitigation", heading_style))
    elements.append(Spacer(1, 10))
    
    # Create a table for post-mitigation metrics
    fairness_data = [['Metric', 'Value', 'Ideal Range', 'Improvement']]
    
    for metric, value in fairness_metrics.items():
        original_metric = metric.replace(" After Mitigation", "")
        improvement = ""
        
        if original_metric in bias_metrics:
            original_value = bias_metrics[original_metric]
            
            # Calculate absolute improvement
            abs_improvement = abs(value) - abs(original_value)
            
            if original_metric == "Disparate Impact":
                # For DI, closer to 1.0 is better
                ideal = 1.0
                original_distance = abs(original_value - ideal)
                new_distance = abs(value - ideal)
                improvement = f"{(original_distance - new_distance):.2f}"
                
                # Add +/- sign
                if original_distance > new_distance:
                    improvement = f"+{improvement}"
                elif original_distance < new_distance:
                    improvement = f"-{improvement}"
            else:
                # For other metrics, closer to 0 is better
                improvement = f"{-abs_improvement:.2f}"
                
                # Add +/- sign
                if abs(value) < abs(original_value):
                    improvement = f"+{improvement}"
                elif abs(value) > abs(original_value):
                    improvement = f"{improvement}"
        
        fairness_data.append([
            metric, 
            f"{value:.2f}", 
            ideal_ranges.get(original_metric, "N/A"),
            improvement
        ])
    
    # Create the table
    fairness_table = Table(fairness_data, colWidths=[250, 100, 100, 70])
    fairness_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(fairness_table)
    elements.append(Spacer(1, 20))
    
    # Recommendations
    elements.append(Paragraph("Recommendations", heading_style))
    elements.append(Spacer(1, 10))
    
    # Generate recommendations based on metrics
    recommendations = []
    
    # Check Disparate Impact
    if 'Disparate Impact After Mitigation' in fairness_metrics:
        di = fairness_metrics['Disparate Impact After Mitigation']
        if di < 0.8 or di > 1.2:
            recommendations.append(
                "The Disparate Impact value is still outside the ideal range (0.8-1.2). "
                "Consider trying a different mitigation technique or adjusting the parameters."
            )
        else:
            recommendations.append(
                "The Disparate Impact value is now within the acceptable range. "
                "This indicates good progress in bias mitigation."
            )
    
    # Check Statistical Parity Difference
    if 'Statistical Parity Difference After Mitigation' in fairness_metrics:
        spd = fairness_metrics['Statistical Parity Difference After Mitigation']
        if abs(spd) > 0.1:
            recommendations.append(
                "The Statistical Parity Difference is still outside the ideal range (-0.1 to 0.1). "
                "This suggests that demographic groups are still receiving positive outcomes at different rates."
            )
        else:
            recommendations.append(
                "The Statistical Parity Difference is now within the acceptable range. "
                "Different demographic groups are receiving positive outcomes at similar rates."
            )
    
    # Check Equal Opportunity Difference
    if 'Equal Opportunity Difference After Mitigation' in fairness_metrics:
        eod = fairness_metrics['Equal Opportunity Difference After Mitigation']
        if abs(eod) > 0.1:
            recommendations.append(
                "The Equal Opportunity Difference is still outside the ideal range. "
                "Consider techniques that specifically target true positive rate differences."
            )
        else:
            recommendations.append(
                "The Equal Opportunity Difference is now within the acceptable range. "
                "The model is providing similar true positive rates across demographic groups."
            )
    
    # Check Average Odds Difference
    if 'Average Odds Difference After Mitigation' in fairness_metrics:
        aod = fairness_metrics['Average Odds Difference After Mitigation']
        if abs(aod) > 0.1:
            recommendations.append(
                "The Average Odds Difference is still outside the ideal range. "
                "This indicates disparities in both true positive and false positive rates."
            )
        else:
            recommendations.append(
                "The Average Odds Difference is now within the acceptable range. "
                "The model is providing similar prediction qualities across demographic groups."
            )
    
    # Add general recommendations
    recommendations.append(
        "Continue monitoring the model for bias during deployment. "
        "Bias can emerge over time as data distributions change."
    )
    
    recommendations.append(
        "Consider collecting additional data from underrepresented groups "
        "to improve model performance for all demographic segments."
    )
    
    # Add recommendations to PDF
    for i, rec in enumerate(recommendations):
        elements.append(Paragraph(f"{i+1}. {rec}", normal_style))
        elements.append(Spacer(1, 5))
    
    # Create the PDF document
    doc.build(elements)
    
    # Return the PDF as bytes
    buffer.seek(0)
    return buffer


def get_table_download_link(df, filename="data.csv", text="Download CSV file"):
    """
    Generates a link allowing the data in a given dataframe to be downloaded
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to be downloaded
    filename : str
        Name of file to be downloaded
    text : str
        Text to display for download link
    
    Returns:
    --------
    str
        HTML code for download link
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def plot_to_base64(fig):
    """
    Convert a matplotlib figure to base64 encoded string for embedding in HTML/PDF
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to convert
    
    Returns:
    --------
    str
        Base64 encoded string of the figure
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def plot_metric_comparison(before_metrics, after_metrics):
    """
    Create a bar chart comparing metrics before and after mitigation
    
    Parameters:
    -----------
    before_metrics : dict
        Dictionary of metrics before mitigation
    after_metrics : dict
        Dictionary of metrics after mitigation
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the comparison chart
    """
    # Set up the metrics for comparison (ensure they match)
    metrics = []
    before_values = []
    after_values = []
    
    for metric, before_value in before_metrics.items():
        after_metric = f"{metric} After Mitigation"
        if after_metric in after_metrics:
            metrics.append(metric)
            before_values.append(before_value)
            after_values.append(after_metrics[after_metric])
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set width of bars
    bar_width = 0.35
    
    # Set position of bars on x-axis
    r1 = np.arange(len(metrics))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    ax.bar(r1, before_values, width=bar_width, label='Before Mitigation', color='indianred')
    ax.bar(r2, after_values, width=bar_width, label='After Mitigation', color='seagreen')
    
    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Fairness Metrics Before and After Mitigation')
    ax.set_xticks([r + bar_width/2 for r in range(len(metrics))])
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    
    # Add ideal range indicators
    for i, metric in enumerate(metrics):
        if metric == "Disparate Impact":
            # For DI, ideal is 0.8-1.2
            ax.axhspan(0.8, 1.2, alpha=0.2, color='green')
        else:
            # For others, ideal is -0.1 to 0.1
            ax.axhspan(-0.1, 0.1, alpha=0.2, color='green')
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def generate_confusion_matrix_fig(cm, classes):
    """
    Generates a confusion matrix visualization
    
    Parameters:
    -----------
    cm : numpy.ndarray
        Confusion matrix values
    classes : list
        List of class names
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the confusion matrix
    """
    # Calculate percentages for each cell
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=classes, yticklabels=classes, ax=ax)
    
    # Add percentage text on top of the counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.25, f"{cm_percentage[i, j]:.1f}%", 
                   ha='center', va='center', color='black', fontsize=10)
    
    # Set labels and title
    ax.set_title("Confusion Matrix", fontsize=16)
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    
    plt.tight_layout()
    
    return fig


def format_run_time(seconds):
    """
    Format runtime in seconds to a readable string
    
    Parameters:
    -----------
    seconds : float
        Runtime in seconds
    
    Returns:
    --------
    str
        Formatted runtime string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)} minutes {int(secs)} seconds"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)} hours {int(minutes)} minutes"


def save_model_info(model_info, filename="model_info.txt"):
    """
    Save model information to a text file
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing model information
    filename : str
        Name of file to save to
    
    Returns:
    --------
    str
        Path to saved file
    """
    with open(filename, 'w') as f:
        f.write("===== MODEL INFORMATION =====\n\n")
        
        # Model type
        f.write(f"Model Type: {model_info.get('model_type', 'Unknown')}\n\n")
        
        # Hyperparameters
        f.write("Hyperparameters:\n")
        for param, value in model_info.get('hyperparameters', {}).items():
            f.write(f"  - {param}: {value}\n")
        f.write("\n")
        
        # Performance metrics
        f.write("Performance Metrics:\n")
        for metric, value in model_info.get('performance', {}).items():
            f.write(f"  - {metric}: {value:.4f}\n")
        f.write("\n")
        
        # Fairness metrics
        f.write("Fairness Metrics:\n")
        for metric, value in model_info.get('fairness', {}).items():
            f.write(f"  - {metric}: {value:.4f}\n")
        f.write("\n")
        
        # Feature importance
        f.write("Feature Importance:\n")
        for feature, importance in model_info.get('feature_importance', {}).items():
            f.write(f"  - {feature}: {importance:.4f}\n")
        
    return filename


def create_summary_table(df, target_col, sensitive_attr):
    """
    Create a summary table of target distribution by sensitive attribute
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to summarize
    target_col : str
        Name of target column
    sensitive_attr : str
        Name of sensitive attribute column
    
    Returns:
    --------
    pandas.DataFrame
        Summary table
    """
    # Create contingency table
    contingency = pd.crosstab(
        df[sensitive_attr], 
        df[target_col], 
        normalize='index',
        margins=True
    ) * 100
    
    # Format as percentages
    formatted = contingency.round(2).astype(str) + '%'
    
    # Add raw counts in parentheses
    counts = pd.crosstab(df[sensitive_attr], df[target_col], margins=True)
    
    for col in counts.columns:
        for idx in counts.index:
            formatted.loc[idx, col] = f"{formatted.loc[idx, col]} ({counts.loc[idx, col]})"
    
    return formatted