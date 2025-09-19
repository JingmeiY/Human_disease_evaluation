"""
Google Forms Creator for SME Evaluation
Creates Google Forms based on the SME evaluation plan with dynamic data population.

QUICK START:
-----------
For Google Colab (easiest):
    creator = GoogleFormsCreator(project_id="your-project-id", use_colab=True)
    form_id = creator.create_simple_test_form("My Test Form")

For local environment with OAuth:
    creator = GoogleFormsCreator(credentials_file="credentials.json", use_colab=False)
    form_id = creator.create_simple_test_form("My Test Form")

Command line usage:
    python google_forms_creator.py --colab --test --project_id="your-project-id"
    python google_forms_creator.py --oauth --test --credentials="credentials.json"
"""

import os
import json
from typing import List, Dict, Any, Optional
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Try to import Colab auth for Colab environments
try:
    from google.colab import auth
    from google.auth import default
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

from utility import load_json_file, save_json_file


class GoogleFormsCreator:
    """
    A class to create Google Forms for SME evaluation using the Google Forms API.
    """
    
    # Scopes required for Google Forms API
    SCOPES = ['https://www.googleapis.com/auth/forms.body']
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.json', 
                 project_id: str = None, use_colab: bool = None):
        """
        Initialize the GoogleFormsCreator.
        
        Args:
            credentials_file (str): Path to Google API credentials JSON file (OAuth)
            token_file (str): Path to store/load authentication token (only used for OAuth)
            project_id (str): Google Cloud Project ID (optional, for Colab)
            use_colab (bool): Force use of Colab authentication. If None, auto-detect
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.project_id = project_id
        self.use_colab = use_colab if use_colab is not None else COLAB_AVAILABLE
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Google API and create the Forms service."""
        if self.use_colab:
            self.authenticate_colab()
        else:
            self.authenticate_oauth()
    
    def authenticate_colab(self):
        """Authenticate using Google Colab authentication."""
        try:
            print("🔐 Authenticating with Google Colab...")
            
            # Set up project ID if provided
            if self.project_id:
                print(f"🔧 Setting up project: {self.project_id}")
                os.environ['GOOGLE_CLOUD_PROJECT'] = self.project_id
                os.environ['GCLOUD_PROJECT'] = self.project_id
            
            # Authenticate
            auth.authenticate_user()
            creds, _ = default()
            
            # Set quota project if available
            if self.project_id and hasattr(creds, 'with_quota_project'):
                creds = creds.with_quota_project(self.project_id)
                print(f"✅ Using project: {self.project_id}")
            
            # Build the Forms API service
            self.service = build('forms', 'v1', credentials=creds)
            print("✅ Google Forms API authentication successful using Colab!")
            
        except Exception as e:
            print(f"❌ Colab authentication failed: {e}")
            print("Make sure you're running in Google Colab and have proper permissions")
            raise
    
    def authenticate_oauth(self):
        """Authenticate using OAuth flow for local environments."""
        print("🔐 Authenticating with OAuth...")
        creds = None
        
        # Load existing token if available
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
        
        # If there are no valid credentials, request authorization
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("🔄 Refreshing expired token...")
                creds.refresh(Request())
            else:
                print("🌐 Starting OAuth flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES)
                
                # Try local server first, fallback to console
                try:
                    print("🖥️ Starting local server for authentication...")
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    print(f"⚠️ Local server authentication failed: {e}")
                    print("🔗 Falling back to console authentication:")
                    print("1. Open the URL that will be displayed")
                    print("2. Complete the authentication in your browser") 
                    print("3. Copy and paste the authorization code back here")
                    creds = flow.run_console()
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
                print(f"💾 Saved credentials to {self.token_file}")
        
        # Build the Forms API service
        self.service = build('forms', 'v1', credentials=creds)
        print("✅ Google Forms API authentication successful using OAuth!")
    
    def create_form_structure(self, title: str, description: str) -> Dict[str, Any]:
        """
        Create the basic form structure for initial creation.
        Note: Google Forms API only allows setting title during creation.
        Description and other settings must be added via batchUpdate.
        
        Args:
            title (str): Form title
            description (str): Form description (will be added later via batchUpdate)
            
        Returns:
            Dict: Minimal form structure for Google Forms API creation
        """
        # Only include title for initial creation
        return {
            "info": {
                "title": title
            }
        }
    
    def create_introduction_section(self) -> List[Dict[str, Any]]:
        """Create the introduction section items."""
        items = []
        
        # Introduction text
        intro_text = """You are evaluating AI-generated outbreak risk assessments for news articles about health threats.

**WHAT YOU'LL SEE:**
- **Article:** News article from HealthMap archived reports of historical disease outbreaks
- **Reference Assessment:** Our current "gold standard" ground truth, advanced AI model output (GPT-o1) with score + reasoning
- **Model A Assessment & Model B Assessment:** Two randomized AI model outputs with scores + reasoning (one fine-tuned, one base model)

## Key Definitions

**Risk Score (1-5):** Outbreak potential rating across multiple risk domains
- **1:** Very low risk
- **2:** Low risk
- **3:** Moderate risk
- **4:** High risk
- **5:** Very high risk

**Reasoning:** justification explaining the score based on relevant factors for each risk domain

## YOUR EVALUATION TASKS

**1. REFERENCE VALIDATION**
- **What it is:** Our current best AI assessment (GPT-o1 advanced risk assessment and reasoning)
- **Your job:** Rate if this "gold standard" is actually accurate and appropriate
- **Why important:** Validates our training baseline

**2. MODEL COMPARISON**
- **What they are:** Two different AI models (one fine-tuned, one base model, order randomized)
- **Your job:** Compare which gives better risk assessment and reasoning

**3. EXPERT ASSESSMENT (OPTIONAL)**
- **Your job:** Provide your own expert risk score if you disagree with all AI assessments
- **When to use:** When you believe all AI models significantly over/under-estimate risk

## Evaluation Focus

**Rate based on:**
- Scientific accuracy
- Appropriateness of score for the described threat level  
- Quality and completeness of reasoning"""
        
        items.append({
            "title": "Instructions and Definitions",
            "description": intro_text,
            "textItem": {}
        })
        
        return items
    
    def create_sample_section(self, sample_data: Dict[str, Any], sample_index: int) -> List[Dict[str, Any]]:
        """
        Create a section for a single sample evaluation.
        
        Args:
            sample_data (Dict): Data for this specific sample
            sample_index (int): Index of the sample (for numbering)
            
        Returns:
            List[Dict]: List of form items for this sample
        """
        items = []
        
        # Sample header
        header_text = f"""═══════════════════════════════════════════════
ARTICLE [{sample_index}]: {sample_data.get('title', 'Sample Article')}
Domain: {sample_data.get('domain', 'Unknown Domain')}
Factors: {sample_data.get('factors_description', 'Evaluation factors for this domain')}
═══════════════════════════════════════════════"""
        
        items.append({
            "title": f"Sample {sample_index} - Article Information",
            "description": header_text,
            "textItem": {}
        })
        
        # Article content (collapsible)
        items.append({
            "title": f"Article Content - Sample {sample_index}",
            "description": sample_data.get('article_content', 'Article content will be populated here'),
            "textItem": {}
        })
        
        # Reference Assessment
        ref_text = f"""---REFERENCE VALIDATION---

REFERENCE ASSESSMENT (Generated by Advanced AI GPT-O1):
Overall Score: {sample_data.get('reference_score', 'X')}
Reasoning: {sample_data.get('reference_reasoning', 'Reference reasoning will be populated here')}

---YOUR EVALUATION---"""
        
        items.append({
            "title": f"Reference Assessment - Sample {sample_index}",
            "description": ref_text,
            "textItem": {}
        })
        
        # Reference Reasoning Quality
        items.append({
            "title": f"1. Reference Reasoning Quality (Required) - Sample {sample_index}",
            "description": "Rate the accuracy and quality of the REFERENCE reasoning:",
            "choiceQuestion": {
                "type": "RADIO",
                "options": [
                    {"value": "Poor (Major errors, inappropriate score)"},
                    {"value": "Average (Acceptable but could be improved)"},
                    {"value": "Good (Accurate with minor issues)"}
                ]
            }
        })
        
        # Reference Score Appropriateness
        items.append({
            "title": f"2. Reference Score Appropriateness (Required) - Sample {sample_index}",
            "description": "Is the reference assessment's risk score appropriate for this scenario?",
            "choiceQuestion": {
                "type": "RADIO",
                "options": [
                    {"value": "Too low (should be higher)"},
                    {"value": "Appropriate (correct level)"},
                    {"value": "Too high (should be lower)"}
                ]
            }
        })
        
        # Model Comparison Section
        model_comparison_text = f"""---MODEL COMPARISON---
MODEL A ASSESSMENT:
Overall Score: {sample_data.get('model_a_score', 'Y')}
Reasoning: {sample_data.get('model_a_reasoning', 'Model A reasoning will be populated here')}

MODEL B ASSESSMENT:
Overall Score: {sample_data.get('model_b_score', 'Z')}
Reasoning: {sample_data.get('model_b_reasoning', 'Model B reasoning will be populated here')}"""
        
        items.append({
            "title": f"Model Assessments - Sample {sample_index}",
            "description": model_comparison_text,
            "textItem": {}
        })
        
        # Model Comparison
        items.append({
            "title": f"3. Model Comparison (Required) - Sample {sample_index}",
            "description": "Which model risk score assessment is MORE ACCURATE for outbreak risk evaluation?",
            "choiceQuestion": {
                "type": "RADIO",
                "options": [
                    {"value": "Model A is clearly better"},
                    {"value": "Model A is slightly better"},
                    {"value": "Both are equivalent"},
                    {"value": "Model B is slightly better"},
                    {"value": "Model B is clearly better"}
                ]
            }
        })
        
        # Reasoning Quality Comparison
        items.append({
            "title": f"4. Reasoning Quality Comparison (Required) - Sample {sample_index}",
            "description": "Which model provides BETTER REASONING and justification?",
            "choiceQuestion": {
                "type": "RADIO",
                "options": [
                    {"value": "Model A reasoning is much better"},
                    {"value": "Model A reasoning is slightly better"},
                    {"value": "Both reasoning are equivalent"},
                    {"value": "Model B reasoning is slightly better"},
                    {"value": "Model B reasoning is much better"}
                ]
            }
        })
        
        # Expert Assessment (Optional)
        items.append({
            "title": f"5. Your Expert Assessment (Optional) - Sample {sample_index}",
            "description": "Based on your expertise, what risk score would YOU assign (1-5)?",
            "choiceQuestion": {
                "type": "RADIO",
                "options": [
                    {"value": "Score 1: Very low risk"},
                    {"value": "Score 2: Low risk"},
                    {"value": "Score 3: Moderate risk"},
                    {"value": "Score 4: High risk"},
                    {"value": "Score 5: Very high risk"}
                ]
            }
        })
        
        # Critical Issues or Comments
        items.append({
            "title": f"6. Critical Issues or Comments (Optional) - Sample {sample_index}",
            "description": "Any critical factors missed by ALL assessments or any comments?",
            "textQuestion": {
                "paragraph": True
            }
        })
        
        return items
    
    def create_sme_evaluation_form(self, samples_data: List[Dict[str, Any]], 
                                   form_title: str = "Outbreak Risk Assessment - Expert Evaluation") -> str:
        """
        Create a complete SME evaluation form with multiple samples.
        
        Args:
            samples_data (List[Dict]): List of sample data to populate the form
            form_title (str): Title for the form
            
        Returns:
            str: Form ID of the created form
        """
        # Step 1: Create basic form with only title
        form_body = self.create_form_structure(
            title=form_title,
            description="Expert evaluation of AI-generated outbreak risk assessments"
        )
        
        # Create the form
        result = self.service.forms().create(body=form_body).execute()
        form_id = result['formId']
        form_url = f"https://docs.google.com/forms/d/{form_id}/edit"
        
        print(f"Created form with ID: {form_id}")
        print(f"Form URL: {form_url}")
        
        # Step 2: Add description and settings via batchUpdate
        description_requests = [
            {
                "updateFormInfo": {
                    "info": {
                        "title": form_title,
                        "description": "Expert evaluation of AI-generated outbreak risk assessments"
                    },
                    "updateMask": "description"
                }
            },
            {
                "updateSettings": {
                    "settings": {
                        "quizSettings": {
                            "isQuiz": False
                        }
                    },
                    "updateMask": "quizSettings"
                }
            }
        ]
        
        # Apply description and settings
        self.service.forms().batchUpdate(
            formId=form_id,
            body={"requests": description_requests}
        ).execute()
        
        # Step 3: Add introduction section
        intro_items = self.create_introduction_section()
        
        # Step 4: Add all sample sections
        all_items = intro_items.copy()
        for i, sample_data in enumerate(samples_data, 1):
            sample_items = self.create_sample_section(sample_data, i)
            all_items.extend(sample_items)
        
        # Step 5: Batch update to add all items
        requests = []
        for i, item in enumerate(all_items):
            requests.append({
                "createItem": {
                    "item": item,
                    "location": {
                        "index": i
                    }
                }
            })
        
        # Apply all items in batches (Google Forms API has limits)
        batch_size = 50  # Conservative batch size
        for i in range(0, len(requests), batch_size):
            batch_requests = requests[i:i + batch_size]
            self.service.forms().batchUpdate(
                formId=form_id,
                body={"requests": batch_requests}
            ).execute()
            print(f"Added items {i+1} to {min(i+batch_size, len(requests))}")
        
        print(f"Successfully added {len(all_items)} items to the form")
        return form_id
    
    def configure_form_settings(self, form_id: str):
        """Configure additional form settings."""
        settings_requests = [
            {
                "updateSettings": {
                    "settings": {
                        "quizSettings": {
                            "isQuiz": False
                        }
                    },
                    "updateMask": "quizSettings.isQuiz"
                }
            }
        ]
        
        update_body = {"requests": settings_requests}
        self.service.forms().batchUpdate(formId=form_id, body=update_body).execute()
        print("Form settings configured successfully")
    
    def get_form_url(self, form_id: str) -> Dict[str, str]:
        """
        Get URLs for the created form.
        
        Args:
            form_id (str): The form ID
            
        Returns:
            Dict: Dictionary containing edit and response URLs
        """
        return {
            "form_id": form_id,
            "edit_url": f"https://docs.google.com/forms/d/{form_id}/edit",
            "response_url": f"https://docs.google.com/forms/d/{form_id}/viewform"
        }

    def create_simple_test_form(self, form_title: str = "Test Form") -> str:
        """
        Create a simple test form to verify API is working.
        
        Args:
            form_title (str): Title for the form
            
        Returns:
            str: Form ID of the created form
        """
        # Step 1: Create form with only title (minimal requirement)
        form_body = {
            "info": {
                "title": form_title
            }
        }
        
        try:
            # Create the form
            result = self.service.forms().create(body=form_body).execute()
            form_id = result['formId']
            
            print(f"✅ Created form with ID: {form_id}")
            
            # Step 2: Add a simple question via batchUpdate
            requests = [
                {
                    "createItem": {
                        "item": {
                            "title": "Test Question",
                            "questionItem": {
                                "question": {
                                    "required": True,
                                    "textQuestion": {
                                        "paragraph": False
                                    }
                                }
                            }
                        },
                        "location": {
                            "index": 0
                        }
                    }
                }
            ]
            
            # Add the question
            self.service.forms().batchUpdate(
                formId=form_id,
                body={"requests": requests}
            ).execute()
            
            print(f"✅ Added test question to form")
            return form_id
            
        except Exception as e:
            print(f"❌ Error in form creation: {e}")
            raise

    def test_api_access(self):
        """
        Test basic API access and permissions by trying to create a minimal form.
        """
        print("🔍 Testing Google Forms API access...")
        
        try:
            # Test 1: Try to create a minimal form (this tests API access and permissions)
            print("   Step 1: Testing form creation...")
            form_body = {
                "info": {
                    "title": "API Test Form - Delete Me"
                }
            }
            
            result = self.service.forms().create(body=form_body).execute()
            form_id = result['formId']
            print(f"   ✅ Form creation successful! Form ID: {form_id}")
            
            # Test 2: Try to get the form (this tests read permissions)
            print("   Step 2: Testing form retrieval...")
            form_data = self.service.forms().get(formId=form_id).execute()
            print(f"   ✅ Form retrieval successful! Title: {form_data['info']['title']}")
            
            # Test 3: Try to add a simple item via batchUpdate
            print("   Step 3: Testing batchUpdate...")
            requests = [{
                "createItem": {
                    "item": {
                        "title": "Test Question",
                        "questionItem": {
                            "question": {
                                "required": False,
                                "textQuestion": {
                                    "paragraph": False
                                }
                            }
                        }
                    },
                    "location": {"index": 0}
                }
            }]
            
            self.service.forms().batchUpdate(
                formId=form_id,
                body={"requests": requests}
            ).execute()
            print("   ✅ BatchUpdate successful!")
            
            print("🎉 All API tests passed! Authentication is working properly.")
            
            # Return the test form info
            return {
                'form_id': form_id,
                'edit_url': f"https://docs.google.com/forms/d/{form_id}/edit",
                'response_url': f"https://docs.google.com/forms/d/{form_id}/viewform"
            }
            
        except Exception as e:
            print(f"❌ Google Forms API access failed: {e}")
            
            error_str = str(e)
            if "403" in error_str or "Permission denied" in error_str:
                print("   🔧 SOLUTION: This is a permissions issue. Try:")
                print("      1. Enable Google Forms API in Google Cloud Console")
                print("      2. Enable Google Drive API in Google Cloud Console") 
                print("      3. Make sure billing is enabled for your project")
                print("      4. Re-authenticate with proper permissions")
                
            elif "404" in error_str:
                print("   🔧 SOLUTION: API not found. Enable Google Forms API in Cloud Console")
                
            elif "500" in error_str:
                print("   🔧 SOLUTION: Server error. This could be:")
                print("      1. Billing not enabled (try enabling billing)")
                print("      2. API quota exceeded (check quotas in Cloud Console)")
                print("      3. Temporary Google server issue (try again later)")
                
            elif "401" in error_str:
                print("   🔧 SOLUTION: Authentication issue. Try re-authenticating")
                
            return None

    def create_minimal_form(self):
        """
        Create the most minimal form possible.
        """
        print("🔧 Creating minimal form...")
        
        # Absolute minimal form structure
        form_body = {
            "info": {
                "title": "Minimal Test Form"
            }
        }
        
        try:
            result = self.service.forms().create(body=form_body).execute()
            form_id = result['formId']
            print(f"✅ Successfully created minimal form: {form_id}")
            return form_id
        except Exception as e:
            print(f"❌ Failed to create minimal form: {e}")
            raise


def load_sample_data(data_file: str) -> List[Dict[str, Any]]:
    """
    Load sample data from JSON file.
    
    Args:
        data_file (str): Path to the JSON file containing sample data
        
    Returns:
        List[Dict]: List of sample data dictionaries
    """
    try:
        return load_json_file(data_file)
    except FileNotFoundError:
        print(f"Data file {data_file} not found. Creating sample data...")
        




def main():
    """Main function to demonstrate form creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create Google Forms for SME Evaluation")
    parser.add_argument("--data_file", type=str, default="sample_data_template.json", 
                       help="JSON file containing sample data")
    parser.add_argument("--credentials", type=str, default="credentials.json",
                       help="Google API OAuth credentials JSON file (not needed for Colab)")
    parser.add_argument("--output", type=str, default="form_info.json",
                       help="Output file for form information")
    parser.add_argument("--form_title", type=str, 
                       default="Outbreak Risk Assessment - Expert Evaluation",
                       help="Title for the Google Form")
    parser.add_argument("--project_id", type=str, default="black-heuristic-463515-f6",
                       help="Google Cloud Project ID (for Colab authentication)")
    parser.add_argument("--colab", action="store_true",
                       help="Force use of Colab authentication")
    parser.add_argument("--oauth", action="store_true",
                       help="Force use of OAuth authentication (local)")
    parser.add_argument("--test", action="store_true",
                       help="Create a simple test form instead of the full SME form")
    parser.add_argument("--diagnose", action="store_true",
                       help="Run diagnostic tests to check API access and permissions")
    
    args = parser.parse_args()
    
    # Determine authentication method
    use_colab = None
    if args.colab:
        use_colab = True
        print("🔧 Using Colab authentication (forced)")
    elif args.oauth:
        use_colab = False
        print("🔧 Using OAuth authentication (forced)")
    else:
        print("🔧 Auto-detecting authentication method...")
    
    # Create form creator instance
    try:
        creator = GoogleFormsCreator(
            credentials_file=args.credentials,
            project_id=args.project_id,
            use_colab=use_colab
        )
        
        if args.diagnose:
            # Run diagnostic tests
            print("🩺 Running diagnostic tests...")
            form_info = creator.test_api_access()
            if form_info:
                print("✅ API access is working, trying minimal form creation...")
                form_id = creator.create_minimal_form()
                print(f"✅ Minimal form created successfully!")
                print(f"Form ID: {form_info['form_id']}")
                print(f"Edit URL: {form_info['edit_url']}")
            else:
                print("❌ API access failed. Check the issues above.")
            return
        
        if args.test:
            # Create simple test form
            print("Creating simple test form...")
            form_id = creator.create_simple_test_form("Test Form - " + args.form_title)
        else:
            # Load sample data and create full form
            samples_data = load_sample_data(args.data_file)
            print(f"Loaded {len(samples_data)} samples")
            
            form_id = creator.create_sme_evaluation_form(
                samples_data=samples_data,
                form_title=args.form_title
            )
        
        # Get form URLs
        form_info = creator.get_form_url(form_id)
        
        # Save form information
        save_json_file(args.output, form_info)
        
        print(f"\n🎉 Form created successfully!")
        print(f"Form ID: {form_info['form_id']}")
        print(f"Edit URL: {form_info['edit_url']}")
        print(f"Response URL: {form_info['response_url']}")
        print(f"Form information saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error creating form: {e}")
        raise


if __name__ == "__main__":
    main()
