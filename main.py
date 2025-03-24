import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from PIL import Image
import pandas as pd
import json
import easyocr
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import gray

# Load environment variables
load_dotenv()
reader = easyocr.Reader(['en'])
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up LangChain Google Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Define data models
class NutrientRecommendation(BaseModel):
    nutrient: str = Field(description="Name of the nutrient")
    current_level: str = Field(description="Current level in patient's report")
    recommended_level: str = Field(description="Recommended level")
    food_sources: List[str] = Field(description="List of food sources rich in this nutrient")
    importance: str = Field(description="Why this nutrient is important")

class DietaryRestriction(BaseModel):
    food_category: str = Field(description="Category of food to restrict")
    reason: str = Field(description="Reason based on medical report")
    alternatives: List[str] = Field(description="Healthy alternatives")

class MealSuggestion(BaseModel):
    meal_type: str = Field(description="Type of meal (breakfast, lunch, dinner, snack)")
    suggested_items: List[str] = Field(description="Suggested food items")
    benefits: List[str] = Field(description="Health benefits related to medical conditions")

class NutritionRecommendation(BaseModel):
    patient_summary: str = Field(description="Summary of patient's health based on report")
    key_health_markers: Dict[str, str] = Field(description="Key health markers and their values")
    nutrient_recommendations: List[NutrientRecommendation] = Field(description="Nutrient-specific recommendations")
    dietary_restrictions: List[DietaryRestriction] = Field(description="Foods to avoid or limit")
    meal_suggestions: List[MealSuggestion] = Field(description="Meal suggestions")
    general_advice: str = Field(description="General nutrition advice")
    disclaimer: str = Field(description="Medical disclaimer")

# Set up the parser
parser = PydanticOutputParser(pydantic_object=NutritionRecommendation)

# Define prompt templates
extraction_prompt = ChatPromptTemplate.from_template("""
You are a medical data analyst. Extract all relevant health markers and their values from this medical report text.
Focus on blood tests, vitamin levels, cholesterol, glucose, liver enzymes, kidney function, etc.

Medical Report Text:
{report_text}

Format your response as a JSON with marker names as keys and their values as a dictionary containing 'value', 'unit', and optionally 'status' and 'reference_range'.
""")

nutrition_prompt = ChatPromptTemplate.from_template("""
You are a nutrition expert and registered dietitian. Based on the following medical report data, 
provide personalized nutrition recommendations for the patient.

Medical Report Data:
{medical_data}

Provide detailed nutrition advice addressing each health marker. Include:
1. A summary of the patient's health based on these markers
2. Specific nutrients they need more or less of
3. Foods they should eat more of and why
4. Foods they should avoid or limit and why
5. Meal suggestions that would help improve their health markers
6. General nutrition advice

Format your response according to these instructions:
{format_instructions}

Always include a disclaimer that this advice is not a replacement for professional medical consultation.
""")
def extract_text_from_image(image):
    try:
        text_list = reader.readtext(image, detail=0)
        extracted_text = " ".join(text_list)
        return extracted_text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return None
    
def analyze_medical_report(report_text):
    try:
        # Create a chain without the Pydantic parser for this step
        chain = extraction_prompt | model
        response = chain.invoke({"report_text": report_text})
        
        # The response should be a string containing JSON
        if hasattr(response, 'content'):
            json_str = response.content
        else:
            json_str = str(response)
            
        # Clean up the response if it contains markdown
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0].strip()
            
        # Parse the JSON string into a Python dictionary
        extracted_data = json.loads(json_str)
        return extracted_data
    except Exception as e:
        st.error(f"Error analyzing medical data: {str(e)}")
        return None

def generate_nutrition_recommendations(medical_data):
    try:
        # Use the Pydantic parser only for the nutrition recommendations
        chain = nutrition_prompt | model | parser
        response = chain.invoke({
            "medical_data": json.dumps(medical_data, indent=2),
            "format_instructions": parser.get_format_instructions()
        })
        return response
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None

def generate_nutrition_recommendations(medical_data):
    try:
        chain = nutrition_prompt | model | parser
        response = chain.invoke({
            "medical_data": json.dumps(medical_data, indent=2),
            "format_instructions": parser.get_format_instructions()
        })
        return response
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return None
def generate_nutrition_pdf(recommendations):
    """
    Generate a professionally formatted PDF from nutrition recommendations
    
    Args:
        recommendations (NutritionRecommendation): The nutrition recommendation object
    
    Returns:
        bytes: PDF file content in memory
    """
    # Create a BytesIO buffer to store PDF
    buffer = io.BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                            rightMargin=72, leftMargin=72, 
                            topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Title'].clone('Title')
    title_style.fontSize = 16
    title_style.textColor = gray
    
    section_style = styles['Heading2'].clone('Section')
    section_style.fontSize = 14
    
    normal_style = styles['Normal'].clone('Normal')
    normal_style.fontSize = 10
    
    # Prepare story (content to be added to PDF)
    story = []
    
    # Title
    story.append(Paragraph("Personalized Nutrition Plan", title_style))
    story.append(Spacer(1, 12))
    
    # Patient Summary
    story.append(Paragraph("Patient Summary", section_style))
    story.append(Paragraph(recommendations.patient_summary, normal_style))
    story.append(Spacer(1, 12))
    
    # Key Health Markers
    story.append(Paragraph("Key Health Markers", section_style))
    for marker, value in recommendations.key_health_markers.items():
        story.append(Paragraph(f"{marker}: {value}", normal_style))
    story.append(Spacer(1, 12))
    
    # Nutrient Recommendations
    story.append(Paragraph("Nutrient Recommendations", section_style))
    for rec in recommendations.nutrient_recommendations:
        story.append(Paragraph(f"{rec.nutrient} - Current Level: {rec.current_level}", normal_style))
        story.append(Paragraph(f"Recommended Level: {rec.recommended_level}", normal_style))
        story.append(Paragraph(f"Importance: {rec.importance}", normal_style))
        story.append(Paragraph("Food Sources:", normal_style))
        for food in rec.food_sources:
            story.append(Paragraph(f"• {food}", normal_style))
        story.append(Spacer(1, 6))
    
    # Dietary Restrictions
    story.append(Paragraph("Foods to Limit or Avoid", section_style))
    for restriction in recommendations.dietary_restrictions:
        story.append(Paragraph(f"{restriction.food_category} - {restriction.reason}", normal_style))
        story.append(Paragraph("Alternatives:", normal_style))
        for alt in restriction.alternatives:
            story.append(Paragraph(f"• {alt}", normal_style))
        story.append(Spacer(1, 6))
    
    # Meal Suggestions
    story.append(Paragraph("Meal Suggestions", section_style))
    for meal in recommendations.meal_suggestions:
        story.append(Paragraph(f"{meal.meal_type.capitalize()} Suggestions", normal_style))
        story.append(Paragraph("Items:", normal_style))
        for item in meal.suggested_items:
            story.append(Paragraph(f"• {item}", normal_style))
        story.append(Paragraph("Benefits:", normal_style))
        for benefit in meal.benefits:
            story.append(Paragraph(f"• {benefit}", normal_style))
        story.append(Spacer(1, 6))
    
    # General Advice
    story.append(Paragraph("General Advice", section_style))
    story.append(Paragraph(recommendations.general_advice, normal_style))
    story.append(Spacer(1, 12))
    
    # Disclaimer
    story.append(Paragraph("Disclaimer", section_style))
    story.append(Paragraph(recommendations.disclaimer, normal_style))
    
    # Build PDF
    doc.build(story)
    
    # Get buffer content
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes
# Streamlit UI remains largely the same, only updating the function calls
st.title("Medical Report Nutrition Advisor")
st.write("Upload a scanned medical report to get personalized nutrition recommendations")

uploaded_file = st.file_uploader("Choose a medical report image", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Report", use_container_width=True)
        
        if st.button("Analyze Report"):
            with st.spinner("Processing medical report..."):
                report_text = extract_text_from_image(image)
                
                if report_text:
                    st.subheader("Extracted Text")
                    with st.expander("View extracted text"):
                        st.text(report_text)
                    
                    medical_data = analyze_medical_report(report_text)
                    
                    if medical_data:
                        st.subheader("Detected Health Markers")
                        st.json(medical_data)
                        
                        with st.spinner("Generating nutrition recommendations..."):
                            recommendations = generate_nutrition_recommendations(medical_data)
                            
                            if recommendations:
                                st.subheader("Your Personalized Nutrition Plan")
                                
                                st.markdown(f"### Patient Summary")
                                st.write(recommendations.patient_summary)
                                
                                st.markdown(f"### Key Health Markers")
                                for marker, value in recommendations.key_health_markers.items():
                                    st.write(f"**{marker}:** {value}")
                                
                                st.markdown(f"### Nutrient Recommendations")
                                for rec in recommendations.nutrient_recommendations:
                                    with st.expander(f"{rec.nutrient} - Current: {rec.current_level}"):
                                        st.write(f"**Recommended Level:** {rec.recommended_level}")
                                        st.write(f"**Importance:** {rec.importance}")
                                        st.write("**Food Sources:**")
                                        for food in rec.food_sources:
                                            st.write(f"- {food}")
                                
                                st.markdown(f"### Foods to Limit or Avoid")
                                for restriction in recommendations.dietary_restrictions:
                                    with st.expander(f"Limit {restriction.food_category}"):
                                        st.write(f"**Reason:** {restriction.reason}")
                                        st.write("**Healthier Alternatives:**")
                                        for alt in restriction.alternatives:
                                            st.write(f"- {alt}")
                                
                                st.markdown(f"### Meal Suggestions")
                                for meal in recommendations.meal_suggestions:
                                    with st.expander(f"{meal.meal_type}"):
                                        st.write("**Suggested Items:**")
                                        for item in meal.suggested_items:
                                            st.write(f"- {item}")
                                        st.write("**Health Benefits:**")
                                        for benefit in meal.benefits:
                                            st.write(f"- {benefit}")
                                
                                st.markdown(f"### General Advice")
                                st.write(recommendations.general_advice)
                                
                                st.markdown(f"### Disclaimer")
                                st.warning(recommendations.disclaimer)
                                
                                st.download_button(
                                    label="Download Nutrition Plan as PDF",
                                    data=generate_nutrition_pdf(recommendations),
                                    file_name="nutrition_plan.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.error("Could not generate nutrition recommendations")
                    else:
                        st.error("Could not analyze medical data from the report")
                else:
                    st.error("Could not extract text from the image")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Sidebar remains the same
with st.sidebar:
    st.title("About")
    st.info("""
    This application analyzes medical reports and provides personalized nutrition recommendations.
    
    **Note:** This tool is for informational purposes only and is not a substitute for professional medical advice.
    Always consult with healthcare providers before making dietary changes based on medical results.
    
    **Privacy Notice:** Your medical report data is processed only for generating recommendations and is not stored.
    """)
    
    st.subheader("How It Works")
    st.write("1. Upload your medical report scan")
    st.write("2. Our AI extracts key health markers")
    st.write("3. The system generates personalized nutrition recommendations")
    st.write("4. Review and download your nutrition plan")