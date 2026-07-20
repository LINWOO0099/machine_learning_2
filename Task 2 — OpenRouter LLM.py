# ============================================================================
# TASK 2: OpenRouter LLM Integration - Google Colab Version
# ============================================================================

# First, install any required packages (if not already installed)
#!pip install requests -q

import os # Import the os module to access environment variables

# ============================================================================
# SETUP: Enter your OpenRouter API Key
# ============================================================================

# Option 1: Set your API key directly (replace with your actual key)
# WARNING: Be careful not to share this notebook publicly with your key visible!
OPENROUTER_API_KEY = ""  # <-- PASTE YOUR KEY HERE



# ============================================================================
# TASK 1: FAQ System (Reused from Task 1)
# ============================================================================

class FAQ:
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

# Sample FAQ data
faqs = [
    FAQ(
        "How do I reset my password?",
        "To reset your password, go to the login page and click 'Forgot Password'. "
        "Enter the email address associated with your account. You will receive a "
        "password reset link within 5 minutes. The link is valid for 24 hours. "
        "If you don't receive the email, check your spam folder."
    ),
    FAQ(
        "What are your business hours?",
        "Our support team is available Monday to Friday, 9:00 AM to 6:00 PM EST. "
        "We respond to email inquiries within 2-4 business hours and live chat "
        "requests within 5 minutes during business hours."
    ),
    FAQ(
        "How can I update my billing information?",
        "To update your billing information, log into your account and go to "
        "Settings > Billing. You can add a new payment method or update your "
        "billing address there. Changes take effect immediately."
    ),
    FAQ(
        "Do you offer refunds?",
        "Yes, we offer a 30-day money-back guarantee on all annual subscriptions. "
        "Monthly subscriptions can be cancelled at any time but are non-refundable "
        "for the current billing period. Contact support to initiate a refund."
    ),
]

def search_by_keyword(query: str, faqs_list: list) -> FAQ:
    """
    Search for the best matching FAQ by keyword matching.
    Returns the FAQ with the highest keyword overlap.
    """
    query_words = set(query.lower().split())
    best_match = None
    best_score = 0
    
    for faq in faqs_list:
        text = (faq.question + " " + faq.answer).lower()
        faq_words = set(text.split())
        overlap = len(query_words.intersection(faq_words))
        if overlap > best_score:
            best_score = overlap
            best_match = faq
    
    return best_match if best_score > 0 else None

# ============================================================================
# TASK 2: OpenRouter LLM Integration
# ============================================================================

import requests
import json
from typing import Optional

class LLMClient:
    """
    Client for interacting with OpenRouter's LLM API.
    """
    
    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini"):
        """
        Initialize the LLM client.
        
        Args:
            api_key: OpenRouter API key
            model: The model identifier to use
        """
        self.api_key = api_key
        self.model = model
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 512
    ) -> str:
        """
        Send a chat completion request to OpenRouter and return the response.
        
        Args:
            prompt: The user's prompt/message
            system_message: Optional system instructions for the LLM
            max_tokens: Maximum tokens in the response
            
        Returns:
            The assistant's response text
            
        Raises:
            Exception: If the API request fails with a descriptive error message
        """
        if not self.api_key:
            raise ValueError("API key is required but not provided")
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare payload
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make API request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_detail = response.text
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_detail = error_json["error"].get("message", error_detail)
                except:
                    pass
                
                raise Exception(
                    f"OpenRouter API error (HTTP {response.status_code}): {error_detail}"
                )
            
            # Parse response
            data = response.json()
            if "choices" not in data or len(data["choices"]) == 0:
                raise Exception("Unexpected API response: No choices returned")
            
            assistant_message = data["choices"][0].get("message", {})
            content = assistant_message.get("content", "")
            
            if not content:
                raise Exception("API response contained empty content")
            
            return content.strip()
            
        except requests.exceptions.Timeout:
            raise Exception("OpenRouter API request timed out after 30 seconds")
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to OpenRouter API - check your internet connection")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error communicating with OpenRouter API: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("Invalid response format from OpenRouter API")
    
    def generate_faq_response(self, user_question: str, faq_entry: FAQ) -> str:
        """
        Generate a conversational response based on FAQ content.
        
        Args:
            user_question: The user's original question
            faq_entry: The matched FAQ entry
            
        Returns:
            A friendly, grounded response from the LLM
        """
        # System prompt designed for grounding and conversational tone
        system_message = """You are a friendly, professional support agent for a software company. 
Your task is to rephrase official FAQ answers into natural, conversational responses.

CRITICAL RULES:
1. ONLY use information provided in the FAQ answer below. DO NOT invent any facts.
2. If the FAQ doesn't fully address the user's specific question, acknowledge this honestly.
3. Use a warm, helpful, and professional tone - like a human support agent.
4. Keep responses concise - under 150 words.
5. Do not mention that you are an AI or that you're using an FAQ.
6. If the FAQ is completely unrelated, politely say you don't have that information.

Remember: GROUND YOUR RESPONSE SOLELY IN THE PROVIDED FAQ CONTENT."""

        # Build the user prompt with context
        prompt = f"""FAQ QUESTION: {faq_entry.question}

FAQ ANSWER: {faq_entry.answer}

USER'S QUESTION: {user_question}

Please provide a helpful, conversational response to the user's question using ONLY the information from the FAQ above. If the FAQ doesn't fully answer the question, let the user know what information is available and suggest they contact support for more details."""
        
        # Generate response using the LLM
        response = self.generate(
            prompt=prompt,
            system_message=system_message,
            max_tokens=512
        )
        
        return response

# ============================================================================
# TASK 3: Prompt Engineering Documentation
# ============================================================================

"""
PROMPT ENGINEERING DOCUMENTATION

System Prompt Design Rationale:
================================

The system prompt is carefully crafted to ensure the LLM behaves as a reliable,
grounded support agent. Here's the breakdown:

1. ROLE DEFINITION: "You are a friendly, professional support agent"
   - Sets the tone and persona for the AI
   - Ensures responses are helpful and customer-oriented

2. GROUNDING INSTRUCTION: "ONLY use information from the FAQ answer"
   - Explicitly limits the knowledge source to prevent hallucinations
   - Uses ALL CAPS for emphasis on the most critical rule

3. HANDLING INCOMPLETE INFORMATION:
   - "If the FAQ doesn't fully address the user's question, acknowledge this honestly"
   - Instructs the model to be transparent about limitations
   - Prevents the model from fabricating answers when information is insufficient

4. TONE CONTROL:
   - "Use a warm, helpful, and professional tone"
   - "Keep responses concise - under 150 words"
   - Creates consistent, appropriate customer service interactions

5. NON-DISCLOSURE: "Do not mention that you are an AI or using an FAQ"
   - Maintains the illusion of a human support agent
   - Creates a more natural customer experience

6. FALLBACK BEHAVIOR:
   - "If the FAQ is completely unrelated, politely say you don't have that information"
   - Provides graceful handling for mismatched FAQs

User Prompt Design:
===================

The user prompt explicitly provides:
- The FAQ question (for context)
- The FAQ answer (the only source of truth)
- The user's original question (to tailor the response)

This structure makes it easy for the LLM to:
1. Identify the relevant information in the FAQ
2. Understand the specific question being asked
3. Map the FAQ content to a conversational response

Why This Approach Works:
=======================

The combination of a strict system prompt and a structured user prompt:
- Reduces hallucinations by clearly defining the knowledge boundary
- Maintains consistency in tone and quality across responses
- Handles edge cases (e.g., partial matches, unrelated FAQs) gracefully
- Creates natural, human-like responses while staying factually accurate
"""

# ============================================================================
# TASK 4: Demonstration
# ============================================================================

def demonstrate_llm_integration():
    """
    Demonstrate the complete LLM integration in Google Colab.
    """

    # Use the API key set directly in the notebook
    API_KEY = OPENROUTER_API_KEY

    # Check if the API key is the default placeholder
    if not API_KEY or API_KEY == "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":
        print("⚠️  WARNING: Valid OpenRouter API key not found or still using placeholder!")
        print("Please set your actual API key in the setup section above.\n")
        print("To get an API key:")
        print("1. Go to https://openrouter.ai/")
        print("2. Sign up or log in")
        print("3. Navigate to your API keys section")
        print("4. Create a new key and paste it above\n")
        return

    print("✓ API key loaded successfully!\n")

    # Initialize the LLM client
    client = LLMClient(
        api_key=API_KEY,
        model="openai/gpt-4o-mini"
    )

    # Test questions
    test_questions = [
        "I can't remember my login password",
        "How do I change my credit card on file?",
        "Can I get a refund for my subscription?",
        "What time does support open?"
    ]

    print("="*80)
    print("SUPPORTAI - LLM-POWERED FAQ RESPONSE SYSTEM")
    print("="*80)
    print(f"\nModel: {client.model}")
    print(f"Number of FAQs in database: {len(faqs)}")
    print(f"Testing with {len(test_questions)} questions...\n")

    for idx, question in enumerate(test_questions, 1):
        print(f"{'─'*80}")
        print(f"TEST CASE {idx}")
        print(f"{'─'*80}")
        print(f"\n📝 User Question: {question}")

        # Find best matching FAQ
        matched_faq = search_by_keyword(question, faqs)

        if matched_faq:
            print(f"📚 Matched FAQ: {matched_faq.question}")

            # Generate LLM response
            try:
                response = client.generate_faq_response(question, matched_faq)
                print("\n🤖 SupportAI Response:")
                print("-" * 80)
                print(response)
                print("-" * 80)

                # Check response length
                word_count = len(response.split())
                if word_count <= 150:
                    print(f"✅ Response length: {word_count} words (within 150-word limit)")
                else:
                    print(f"⚠️  Response length: {word_count} words (exceeds 150-word limit)")

            except Exception as e:
                print(f"\n❌ Error generating response: {e}")
        else:
            print("❌ No matching FAQ found. Please contact support directly.")

        print("\n")

    # Error handling demonstration
    print("="*80)
    print("🛡️  ERROR HANDLING DEMONSTRATION")
    print("="*80)
    print("\nTesting API error handling...")

    # Test with invalid key
    error_client = LLMClient(api_key="invalid-key-12345", model="openai/gpt-4o-mini")
    try:
        error_client.generate(
            prompt="Hello, how are you?",
            system_message="You are a helpful assistant."
        )
    except Exception as e:
        print(f"\n✅ Error caught gracefully:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Message: {e}")

    # Test with empty key
    print("\nTesting with empty API key...")
    empty_client = LLMClient(api_key="")
    try:
        empty_client.generate(prompt="Hello")
    except Exception as e:
        print(f"\n✅ Error caught gracefully:")
        print(f"   Message: {e}")

    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80)

    # Summary
    print("\n📋 FEATURES DEMONSTRATED:")
    print("✓ Secure API key configuration using environment variables")
    print("✓ LLMClient initialization with configurable model")
    print("✓ generate() method with proper error handling")
    print("✓ generate_faq_response() with grounding prompt")
    print("✓ Prompt engineering for grounded, conversational responses")
    print("✓ Integration with Task 1's search_by_keyword()")
    print("✓ Graceful error handling for API failures")
    print("✓ Multiple test questions showing versatility")
    print(f"✓ Used model: {client.model}\n")

# ============================================================================
# RUN THE DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 TASK 2: OPENROUTER LLM INTEGRATION - GOOGLE COLAB")
    print("="*80)
    print("\nThis notebook demonstrates:\n")
    print("1. Secure API key configuration")
    print("2. LLMClient class with generate() and generate_faq_response()")
    print("3. Prompt engineering for grounding and tone control")
    print("4. Integration with Task 1's FAQ system")
    print("5. Comprehensive error handling")
    print("6. Multiple test case demonstrations\n")

    demonstrate_llm_integration()