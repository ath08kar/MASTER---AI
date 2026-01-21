# agents/report_writer.py

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from datetime import datetime
import os

def generate_explanation_html(symbol, date, market_output, tech_probs, final_decision):
    """
    Creates the HTML-formatted text block for the report.
    """
    # Unpack probabilities safely
    try:
        p_down, p_hold, p_up = tech_probs
    except ValueError:
        p_down, p_hold, p_up = 0.0, 0.0, 0.0

    # Format the decision color
    decision_color = "green" if final_decision['decision'] == "BUY" else "red" if final_decision['decision'] == "SELL" else "black"

    explanation = f"""
    <b>Stock:</b> {symbol}<br/>
    <b>Date:</b> {date}<br/><br/>

    <font size="12"><b>1. Market Context</b></font><br/>
    The market regime is identified as <b>{market_output.get('regime', 'Unknown')}</b> 
    with a confidence score of <b>{market_output.get('confidence', 0):.2f}</b>.<br/>
    <i>Volatility Level: {market_output.get('volatility', 'N/A')}</i><br/><br/>

    <font size="12"><b>2. Technical Model Output</b></font><br/>
    The neural network estimates the following probabilities:<br/>
    &nbsp;&nbsp;• <b>UP:</b> {p_up:.1%}<br/>
    &nbsp;&nbsp;• <b>HOLD:</b> {p_hold:.1%}<br/>
    &nbsp;&nbsp;• <b>DOWN:</b> {p_down:.1%}<br/><br/>

    <font size="12"><b>3. Final Decision</b></font><br/>
    The orchestrator recommends: <font color="{decision_color}"><b>{final_decision['decision']}</b></font><br/>
    <b>Rationale:</b> {final_decision['reason']}
    """
    return explanation

def generate_pdf_report(filename, data, chart_path=None):
    """
    Main function to build the PDF.
    
    Args:
        filename (str): Path to save PDF.
        data (dict): The session_state data dictionary from app.py.
        chart_path (str): Path to the temporary chart image.
    """
    
    doc = SimpleDocTemplate(filename, pagesize=LETTER)
    styles = getSampleStyleSheet()

    # --- Custom Styles ---
    title_style = ParagraphStyle(
        "ReportTitle", 
        parent=styles['Heading1'], 
        fontSize=24, 
        alignment=1, 
        textColor=colors.darkblue,
        spaceAfter=20
    )
    
    body_style = ParagraphStyle(
        "ReportBody", 
        parent=styles['Normal'], 
        fontSize=11, 
        leading=16, 
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        "Meta",
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=1
    )

    elements = []

    # --- 1. COVER PAGE ---
    elements.append(Spacer(1, 100))
    elements.append(Paragraph(f"AI Alpha Report", title_style))
    elements.append(Paragraph(f"Analysis Target: {data['ticker']}", ParagraphStyle('Sub', parent=title_style, fontSize=18, textColor=colors.black)))
    elements.append(Spacer(1, 50))
    
    # Timestamp
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
    elements.append(PageBreak())

    # --- 2. ANALYSIS PAGE ---
    elements.append(Paragraph("Executive Summary", title_style))
    
    # Generate the text content
    content_html = generate_explanation_html(
        symbol=data['ticker'],
        date=datetime.now().strftime('%Y-%m-%d'),
        market_output=data['market_out'],
        tech_probs=data['probs'],
        final_decision=data['decision']
    )
    
    elements.append(Paragraph(content_html, body_style))
    elements.append(Spacer(1, 20))

    # --- 3. CHART INTEGRATION ---
    if chart_path and os.path.exists(chart_path):
        elements.append(Paragraph("Price Action Analysis", styles['Heading2']))
        elements.append(Spacer(1, 10))
        try:
            # Aspect ratio usually 16:9, width=500 is good for Letter page
            img = RLImage(chart_path, width=500, height=300) 
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"[Chart Image Missing: {e}]", body_style))
            
    doc.build(elements)
    return filename