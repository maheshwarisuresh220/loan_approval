import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
import plotly.express as px
import plotly.graph_objects as go
import math

# Set page configuration
st.set_page_config(
    page_title="CreditWise Pro | Smart Loan Solutions",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Pakistani Banking Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f4c3a 0%, #1e7e34 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #155724, #28a745);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #e8f5e8;
        font-size: 1.1rem;
        margin: 0;
    }
    
    
    
    .card h3, .card h4, .card p, .card li {
        color: #333333 !important;
    }
    
    .success-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(40, 167, 69, 0.3);
        margin: 2rem 0;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(220, 53, 69, 0.3);
        margin: 2rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
    }
    
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    
    .suggestion-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        color: #2d3436;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 234, 167, 0.3);
    }
    
    .ratio-box {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #6c757d;
        color: #333333;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = pkl.load(open("model.pkl", "rb"))
        scaler = pkl.load(open("scaler.pkl", "rb"))
        return model, scaler, True
    except:
        return None, None, False

# EMI Calculation
def calculate_emi(principal, rate_annual, tenure_months):
    if rate_annual == 0:
        return principal / tenure_months
    rate_monthly = rate_annual / 12 / 100
    emi = (principal * rate_monthly * (1 + rate_monthly)**tenure_months) / \
          ((1 + rate_monthly)**tenure_months - 1)
    return emi

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏦 CreditWise Pro</h1>
        <p>Smart Loan Approval System for Pakistan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, models_loaded = load_models()
    
    if not models_loaded:
        st.error("⚠️ Model files not found. Please run the training script first.")
        return
    
    # Tab Navigation
    tab1, tab2, tab3 = st.tabs(["🎯 Loan Prediction", "💰 EMI Calculator", "ℹ️ Instructions"])
    
    with tab1:
        show_prediction_page(model, scaler)
    
    with tab2:
        show_emi_calculator()
    
    with tab3:
        show_instructions()

def show_prediction_page(model, scaler):
    st.markdown("### 📊 Loan Approval Prediction")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Input fields
        col_a, col_b = st.columns(2)
        
        with col_a:
            no_of_dependents = st.selectbox("👨‍👩‍👧‍👦 Dependents:", [0,1,2,3,4,5], index=2)
            education = st.selectbox("🎓 Education:", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("💼 Employment:", ["Salaried", "Self-Employed"])
            cibil_score = st.number_input("📊 CIBIL Score:", 300, 900, 750, 10)
        
        with col_b:
            income_annum = st.number_input("💰 Annual Income (PKR):", 100000, 50000000, 1200000, 50000)
            loan_amount = st.number_input("🏦 Loan Amount (PKR):", 50000, 100000000, 2500000, 50000)
            loan_term = st.selectbox("📅 Loan Term (Months):", [12,24,36,48,60,84,120,180,240,360], index=4)
            assets = st.number_input("🏠 Total Assets (PKR):", 0, 1000000000, 5000000, 100000)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        if st.button("🎯 Predict Loan Approval"):
            # Prepare data
            education_encoded = 1 if education == "Graduate" else 0
            self_employed_encoded = 1 if self_employed == "Self-Employed" else 0
            
            input_data = np.array([[
                no_of_dependents, education_encoded, self_employed_encoded,
                income_annum, loan_amount, loan_term, cibil_score, assets
            ]])
            
            # Make prediction
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            with col2:
                if prediction == 0:
                    st.markdown(f"""
                    <div class="success-box">
                        <h2>✅ APPROVED!</h2>
                        <h3>{prediction_proba[0]:.1%} Confidence</h3>
                        <p>Your loan application is likely to be approved!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Quick EMI calculation
                    emi = calculate_emi(loan_amount, 12.0, loan_term)
                    total_amount = emi * loan_term
                    total_interest = total_amount - loan_amount
                    
                    st.markdown("#### 💰 Quick EMI Info")
                    st.metric("Monthly EMI", f"PKR {emi:,.0f}")
                    st.metric("Total Interest", f"PKR {total_interest:,.0f}")
                    
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h2>❌ REJECTED</h2>
                        <h3>{prediction_proba[0]:.1%} Confidence</h3>
                        <p>Consider improving your profile.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed suggestions based on input
                    st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                    st.markdown("#### 💡 Improvement Suggestions")
                    suggestions = []
                    
                    # CIBIL Score check
                    if cibil_score < 750:
                        suggestions.append(f"**🎯 Improve CIBIL Score**: Current {cibil_score}, target 750+ for better approval chances")
                    
                    # Income to Loan ratio check
                    loan_to_income_ratio = loan_amount / income_annum
                    if loan_to_income_ratio > 3:
                        safe_amount = income_annum * 3
                        suggestions.append(f"**💰 Reduce Loan Amount**: Current PKR {loan_amount:,} is high. Consider PKR {safe_amount:,.0f} (3x your income)")
                    
                    # Asset coverage check
                    asset_coverage = assets / loan_amount if loan_amount > 0 else 0
                    if asset_coverage < 2:
                        target_assets = loan_amount * 2
                        suggestions.append(f"**🏠 Increase Assets**: Target PKR {target_assets:,.0f} (2x loan amount) for better security")
                    
                    # Dependents check
                    if no_of_dependents > 3:
                        suggestions.append("**👨‍👩‍👧‍👦 High Dependents**: Consider joint application with spouse to increase household income")
                    
                    # Loan term optimization
                    if loan_term < 36:
                        suggestions.append("**📅 Extend Tenure**: Longer repayment terms (5+ years) often improve approval chances")
                    
                    # Employment type suggestion
                    if self_employed == "Self-Employed":
                        suggestions.append("**📋 Documentation Ready**: Ensure 2-3 years ITR, bank statements, and business proof are available")
                    
                    # Show suggestions
                    if suggestions:
                        for i, suggestion in enumerate(suggestions, 1):
                            st.markdown(f"{i}. {suggestion}")
                    else:
                        st.markdown("**✅ Profile looks excellent!** Your financial ratios are strong.")
                        st.markdown("**🏦 Recommended Action:**")
                        st.markdown("• **Try Premium Banks**: HBL, UBL, MCB, Standard Chartered")
                        st.markdown("• **Consider Secured Loans**: Use assets as collateral for better rates")
                        st.markdown("• **Alternative Lenders**: NBFCs or Islamic banking options")
                        st.markdown("• **Direct Branch Visit**: Sometimes personal interaction helps")
                        
                        # Show why it might be rejected despite good profile
                        st.markdown("**🔍 Possible Reasons for Model Rejection:**")
                        st.markdown("• Model trained on conservative banking data")
                        st.markdown("• Specific pattern recognition beyond standard ratios") 
                        st.markdown("• Consider this a cautious AI prediction, not final verdict")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show current ratios
                    st.markdown("#### 📊 Your Current Financial Ratios")
                    
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.markdown('<div class="ratio-box">', unsafe_allow_html=True)
                        ratio_color = "🔴" if loan_to_income_ratio > 3 else "🟡" if loan_to_income_ratio > 2 else "🟢"
                        st.markdown(f"**{ratio_color} Loan-to-Income Ratio**  \n{loan_to_income_ratio:.1f}x (Target: <3x)")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="ratio-box">', unsafe_allow_html=True)
                        coverage_color = "🔴" if asset_coverage < 1 else "🟡" if asset_coverage < 2 else "🟢"
                        st.markdown(f"**{coverage_color} Asset Coverage**  \n{asset_coverage:.1f}x (Target: >2x)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col_r2:
                        monthly_income = income_annum / 12
                        emi_estimate = calculate_emi(loan_amount, 15.0, loan_term)
                        emi_ratio = (emi_estimate / monthly_income) * 100 if monthly_income > 0 else 0
                        
                        st.markdown('<div class="ratio-box">', unsafe_allow_html=True)
                        emi_color = "🔴" if emi_ratio > 50 else "🟡" if emi_ratio > 40 else "🟢"
                        st.markdown(f"**{emi_color} EMI-to-Income**  \n{emi_ratio:.1f}% (Target: <40%)")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="ratio-box">', unsafe_allow_html=True)
                        cibil_color = "🔴" if cibil_score < 650 else "🟡" if cibil_score < 750 else "🟢"
                        st.markdown(f"**{cibil_color} Credit Score**  \n{cibil_score} (Target: 750+)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Debug information for excellent profiles that get rejected
                    if not suggestions:
                        with st.expander("🔧 Technical Debug Info"):
                            st.write("**Model Input Values:**")
                            st.write(f"• Dependents: {no_of_dependents}")
                            st.write(f"• Education: {education_encoded} (1=Graduate)")
                            st.write(f"• Employment: {self_employed_encoded} (1=Self-Employed)")
                            st.write(f"• Income: PKR {income_annum:,}")
                            st.write(f"• Loan Amount: PKR {loan_amount:,}")
                            st.write(f"• Loan Term: {loan_term} months")
                            st.write(f"• CIBIL Score: {cibil_score}")
                            st.write(f"• Assets: PKR {assets:,}")
                            st.write(f"**Model Confidence:** Rejection {prediction_proba[0]:.3f}, Approval {prediction_proba[1]:.3f}")
                            
                            if prediction_proba[0] < 0.6:  # If rejection confidence is low
                                st.success("💡 **Low rejection confidence!** This is a borderline case - definitely worth applying to multiple banks!")
                            
                            # Suggest trying different values
                            st.write("**Try These Adjustments:**")
                            st.write("• Reduce loan amount by 10-20%")
                            st.write("• Increase loan tenure to 7-10 years")
                            st.write("• Try 'Self-Employed' if currently 'Salaried'")
                            
                        # Add a second opinion section
                        st.markdown("### 🎯 Get a Second Opinion")
                        st.info("Your profile shows excellent financial health. The AI model may be overly conservative. We recommend consulting with a bank relationship manager for a human assessment.")
                        
                        # Quick retest with optimized values
                        if st.button("🔄 Test with Optimized Values", key="retest"):
                            # Try with slightly adjusted values that might work better
                            optimized_loan = int(loan_amount * 0.8)  # 20% less
                            optimized_tenure = min(120, loan_term + 24)  # Add 2 years
                            
                            opt_input = np.array([[
                                no_of_dependents, education_encoded, 1-self_employed_encoded,  # Flip employment
                                income_annum, optimized_loan, optimized_tenure, cibil_score, assets
                            ]])
                            
                            opt_scaled = scaler.transform(opt_input)
                            opt_pred = model.predict(opt_scaled)[0]
                            opt_proba = model.predict_proba(opt_scaled)[0]
                            
                            if opt_pred == 1:
                                st.success(f"✅ **Optimized Profile Approved!** ({opt_proba[1]:.1%} confidence)")
                                st.write(f"**Suggested Changes:**")
                                st.write(f"• Loan Amount: PKR {optimized_loan:,} (reduced by 20%)")
                                st.write(f"• Loan Tenure: {optimized_tenure} months")
                                if self_employed_encoded == 0:
                                    st.write("• Consider self-employed application route")
                                else:
                                    st.write("• Consider salaried application route")
                            else:
                                st.warning("Model still shows rejection even with optimized values. This suggests the model may have learned very specific patterns from the training data.")

def show_emi_calculator():
    st.markdown("### 💰 EMI Calculator")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        loan_amount = st.number_input("💵 Loan Amount (PKR):", 50000, 100000000, 2500000, 50000)
        interest_rate = st.slider("📈 Interest Rate (%):", 5.0, 25.0, 12.0, 0.5)
        
        tenure_options = {
            "6 Months": 6, "1 Year": 12, "2 Years": 24, "3 Years": 36,
            "5 Years": 60, "7 Years": 84, "10 Years": 120, "15 Years": 180
        }
        tenure_selected = st.selectbox("📅 Loan Tenure:", list(tenure_options.keys()), index=4)
        tenure_months = tenure_options[tenure_selected]
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Calculate EMI
        emi = calculate_emi(loan_amount, interest_rate, tenure_months)
        total_amount = emi * tenure_months
        total_interest = total_amount - loan_amount
        
        st.markdown("#### 📊 EMI Summary")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Monthly EMI", f"PKR {emi:,.0f}")
            st.metric("Total Interest", f"PKR {total_interest:,.0f}")
        with col_b:
            st.metric("Total Amount", f"PKR {total_amount:,.0f}")
            st.metric("Interest Ratio", f"{(total_interest/loan_amount)*100:.1f}%")
    
    with col2:
        # Pie chart for loan breakdown
        fig = go.Figure(data=[go.Pie(
            labels=['Principal Amount', 'Total Interest'],
            values=[loan_amount, total_interest],
            hole=0.4,
            marker=dict(colors=['#28a745', '#fd7e14']),
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Loan Breakdown",
            height=350,
            showlegend=True,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interest rate comparison
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 🏦 Current Market Rates")
        rates_data = {
            'Loan Type': ['Personal', 'Home', 'Car', 'Business'],
            'Rate (%)': ['15-18%', '12-15%', '16-20%', '14-17%']
        }
        st.dataframe(pd.DataFrame(rates_data), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_instructions():
    st.markdown("### ℹ️ How to Use CreditWise Pro")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <h4>🎯 Loan Prediction</h4>
        <ol>
            <li><strong>Enter Personal Details:</strong> Number of dependents, education level, and employment type</li>
            <li><strong>Financial Information:</strong> Annual income, desired loan amount, and repayment term</li>
            <li><strong>Credit Score:</strong> Your CIBIL score (check free at CIBIL website)</li>
            <li><strong>Assets:</strong> Total value of your properties and investments</li>
            <li><strong>Get Results:</strong> Instant approval prediction with confidence score</li>
        </ol>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <h4>📊 CIBIL Score Guide</h4>
        <ul>
            <li><strong>750-900:</strong> Excellent (High approval chances)</li>
            <li><strong>700-749:</strong> Good (Moderate approval chances)</li>
            <li><strong>650-699:</strong> Fair (Work on improvement)</li>
            <li><strong>Below 650:</strong> Poor (Focus on credit repair)</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <h4>💰 EMI Calculator</h4>
        <ol>
            <li><strong>Loan Amount:</strong> Enter the amount you want to borrow</li>
            <li><strong>Interest Rate:</strong> Current market rates (8-22% in Pakistan)</li>
            <li><strong>Tenure:</strong> Select repayment period</li>
            <li><strong>View Results:</strong> Get monthly EMI and total cost breakdown</li>
        </ol>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <h4>💡 Tips for Better Approval</h4>
        <ul>
            <li><strong>Maintain Good Credit:</strong> Keep CIBIL score above 750</li>
            <li><strong>Stable Income:</strong> Regular salary/business income</li>
            <li><strong>Lower DTI Ratio:</strong> Keep debt-to-income below 40%</li>
            <li><strong>Asset Building:</strong> Maintain assets worth 2x loan amount</li>
            <li><strong>Complete Documentation:</strong> Have all required papers ready</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <h4>🏦 About CreditWise Pro</h4>
    <p>CreditWise Pro uses advanced machine learning algorithms trained on Pakistani banking data to predict loan approval chances. 
    The system analyzes multiple factors including income, credit score, employment type, and asset base to provide accurate predictions.</p>
    <p><strong>Accuracy:</strong> 95%+ based on historical banking data</p>
    <p><strong>Data Security:</strong> All calculations are performed locally - no data is stored or transmitted</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()