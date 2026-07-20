# ============================================================================
# TASK 4: COMPLETE HELPDESK AGENT - ADVANCED VERSION (FIXED)
# ============================================================================

import os
import random
import requests
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import re
import logging
from enum import Enum
from collections import defaultdict
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('supportai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TASK 1: FAQ SYSTEM (ENHANCED)
# ============================================================================

class FAQCategory(Enum):
    """FAQ categories for better organization."""
    ACCOUNT = "account"
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"
    GENERAL = "general"
    SECURITY = "security"

@dataclass
class FAQ:
    """Enhanced FAQ class with additional metadata."""
    question: str
    answer: str
    keywords: List[str] = field(default_factory=list)
    faq_id: str = None
    category: FAQCategory = FAQCategory.GENERAL
    priority: int = 1  # 1-5, where 5 is highest
    version: str = "1.0"
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    related_faqs: List[str] = field(default_factory=list)
    view_count: int = 0
    helpful_count: int = 0

    def __post_init__(self):
        if self.faq_id is None:
            self.faq_id = f"faq-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

    def get_combined_text(self) -> str:
        """Get combined text for searching."""
        return f"{self.question} {' '.join(self.keywords)} {' '.join(self.tags)}"

    def increment_views(self):
        self.view_count += 1

    def increment_helpful(self):
        self.helpful_count += 1

# Comprehensive FAQ Database
faqs = [
    FAQ(
        question="How do I reset my password?",
        answer="To reset your password, go to the login page and click 'Forgot Password'. "
               "Enter the email address associated with your account. You will receive a "
               "password reset link within 5 minutes. The link is valid for 24 hours. "
               "If you don't receive the email, check your spam folder.",
        keywords=["password", "reset", "login", "forgot", "credentials", "account", "email"],
        category=FAQCategory.SECURITY,
        priority=5,
        tags=["password", "security", "login", "reset"],
        related_faqs=["faq-006"]
    ),
    FAQ(
        question="What are your business hours?",
        answer="Our support team is available Monday to Friday, 9:00 AM to 6:00 PM EST. "
               "We respond to email inquiries within 2-4 business hours and live chat "
               "requests within 5 minutes during business hours.",
        keywords=["hours", "business", "support", "available", "time", "schedule"],
        category=FAQCategory.GENERAL,
        priority=3,
        tags=["hours", "support", "availability"],
        related_faqs=[]
    ),
    FAQ(
        question="How can I update my billing information?",
        answer="To update your billing information, log into your account and go to "
               "Settings > Billing. You can add a new payment method or update your "
               "billing address there. Changes take effect immediately.",
        keywords=["billing", "payment", "credit card", "update", "change", "address"],
        category=FAQCategory.BILLING,
        priority=4,
        tags=["billing", "payment", "credit card"],
        related_faqs=["faq-004"]
    ),
    FAQ(
        question="Do you offer refunds?",
        answer="Yes, we offer a 30-day money-back guarantee on all annual subscriptions. "
               "Monthly subscriptions can be cancelled at any time but are non-refundable "
               "for the current billing period. Contact support to initiate a refund.",
        keywords=["refund", "money back", "cancel", "subscription", "guarantee", "return"],
        category=FAQCategory.BILLING,
        priority=5,
        tags=["refund", "cancellation", "guarantee"],
        related_faqs=["faq-003"]
    ),
    FAQ(
        question="What is your shipping policy?",
        answer="We offer free standard shipping on all orders over $50. Standard shipping "
               "takes 3-5 business days. Express shipping (2-3 business days) is available "
               "for $15. International shipping takes 7-14 business days.",
        keywords=["shipping", "delivery", "package", "order", "tracking", "delivery time"],
        category=FAQCategory.SHIPPING,
        priority=3,
        tags=["shipping", "delivery", "international"],
        related_faqs=[]
    ),
    FAQ(
        question="How do I update my email address?",
        answer="To update your email address, go to Account Settings > Profile. Click on "
               "the email field, enter your new email, and confirm by clicking 'Save Changes'. "
               "You will receive a verification email to confirm the change.",
        keywords=["email", "update", "change", "profile", "account", "verification"],
        category=FAQCategory.ACCOUNT,
        priority=3,
        tags=["email", "profile", "verification"],
        related_faqs=["faq-001"]
    ),
]

# ============================================================================
# TASK 1: ENHANCED KEYWORD SEARCH
# ============================================================================

class KeywordSearch:
    """Enhanced keyword search with advanced scoring."""

    @staticmethod
    def search(query: str, faqs_list: List[FAQ]) -> Optional[FAQ]:
        """Simple search returning best match."""
        results = KeywordSearch.search_with_score(query, faqs_list)
        return results[0][0] if results else None

    @staticmethod
    def search_with_score(query: str, faqs_list: List[FAQ], top_k: int = 5) -> List[Tuple[FAQ, float]]:
        """Advanced keyword search with weighted scoring."""
        query_words = set(query.lower().split())
        results = []

        for faq in faqs_list:
            # Weighted text combination
            text_question = faq.question.lower()
            text_keywords = " ".join(faq.keywords).lower()
            text_tags = " ".join(faq.tags).lower()

            # Different weights for different fields
            question_words = set(text_question.split())
            keyword_words = set(text_keywords.split())
            tag_words = set(text_tags.split())

            # Calculate weighted overlap
            question_overlap = len(query_words.intersection(question_words))
            keyword_overlap = len(query_words.intersection(keyword_words))
            tag_overlap = len(query_words.intersection(tag_words))

            # Weighted score
            weighted_score = (
                question_overlap * 1.0 +  # Highest weight for question
                keyword_overlap * 0.7 +   # Moderate weight for keywords
                tag_overlap * 0.5         # Lower weight for tags
            )

            if weighted_score > 0:
                # Normalize score
                max_possible = len(query_words) * 1.0
                score = min(weighted_score / max_possible, 1.0)
                results.append((faq, round(score, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ============================================================================
# TASK 2: LLM CLIENT (ENHANCED WITH CACHING)
# ============================================================================

class CacheEntry:
    """Cache entry with timestamp."""
    def __init__(self, response: str, timestamp: float):
        self.response = response
        self.timestamp = timestamp
        self.hits = 0

class LLMClient:
    """Enhanced LLM client with caching, retries, and error handling."""

    def __init__(self, api_key: str, model: str = "openai/gpt-4o-mini", cache_ttl: int = 3600):
        self.api_key = api_key
        self.model = model
        self.api_url = ""
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_ttl = cache_ttl
        self.max_retries = 3
        self.retry_delay = 1
        self.stats = {
            "total_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }

    def _get_cache_key(self, prompt: str, system_message: Optional[str]) -> str:
        """Generate cache key from prompt and system message."""
        return f"{system_message or ''}:{prompt}"

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if valid."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if time.time() - entry.timestamp < self.cache_ttl:
                entry.hits += 1
                self.stats["cache_hits"] += 1
                logger.debug(f"Cache hit for key: {cache_key[:50]}...")
                return entry.response
            else:
                del self.cache[cache_key]
        self.stats["cache_misses"] += 1
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache a response."""
        self.cache[cache_key] = CacheEntry(response, time.time())

    def generate(self, prompt: str, system_message: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate response with caching and retries.
        """
        self.stats["total_calls"] += 1

        # Check cache
        cache_key = self._get_cache_key(prompt, system_message)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response

        # Prepare messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        if "error" in error_json:
                            error_detail = error_json["error"].get("message", error_detail)
                    except:
                        pass
                    raise Exception(f"API error (HTTP {response.status_code}): {error_detail}")

                data = response.json()
                if "choices" not in data or len(data["choices"]) == 0:
                    raise Exception("Unexpected API response: No choices returned")

                assistant_message = data["choices"][0].get("message", {})
                content = assistant_message.get("content", "")

                if not content:
                    raise Exception("API response contained empty content")

                # Cache successful response
                self._cache_response(cache_key, content)
                return content.strip()

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise Exception("OpenRouter API request timed out after multiple retries")
                time.sleep(self.retry_delay * (attempt + 1))

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise Exception("Could not connect to OpenRouter API")
                time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Error in API call: {str(e)}")
                raise

        raise Exception("Max retries exceeded")

    def generate_faq_response(self, user_question: str, faq_entry: FAQ) -> str:
        """Generate FAQ response with enhanced prompt."""
        system_message = """You are a friendly, professional support agent for a software company.
Your task is to rephrase official FAQ answers into natural, conversational responses.

CRITICAL RULES:
1. ONLY use information provided in the FAQ answer below. DO NOT invent any facts.
2. If the FAQ doesn't fully address the user's specific question, acknowledge this honestly.
3. Use a warm, helpful, and professional tone - like a human support agent.
4. Keep responses concise - under 150 words.
5. Do not mention that you are an AI or that you're using an FAQ.
6. If the FAQ is completely unrelated, politely say you don't have that information.
7. Use empathy and understanding in your responses.
8. Offer additional help if appropriate.

Remember: GROUND YOUR RESPONSE SOLELY IN THE PROVIDED FAQ CONTENT."""

        prompt = f"""FAQ QUESTION: {faq_entry.question}

FAQ ANSWER: {faq_entry.answer}

FAQ CATEGORY: {faq_entry.category.value}

USER'S QUESTION: {user_question}

Please provide a helpful, conversational response to the user's question using ONLY the information from the FAQ above. If the FAQ doesn't fully answer the question, let the user know what information is available and suggest they contact support for more details."""

        return self.generate(prompt=prompt, system_message=system_message, max_tokens=512)

# ============================================================================
# TASK 3: FAQ MATCHER (ENHANCED)
# ============================================================================

class FAQMatcher:
    """Enhanced FAQ matcher with multiple strategies."""

    def __init__(self, faqs: List[FAQ]):
        self.faqs = faqs
        self.corpus = self._build_corpus()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),  # Use up to trigrams
            max_df=0.85,
            min_df=1,
            sublinear_tf=True  # Use sublinear TF scaling
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.vectorizer_question = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.question_matrix = self.vectorizer_question.fit_transform(
            [faq.question for faq in faqs]
        )

    def _build_corpus(self) -> List[str]:
        """Build enhanced corpus with weighted fields."""
        corpus = []
        for faq in self.faqs:
            # Repeat question for higher weight
            text = f"{faq.question} {faq.question} "  # Double question
            text += f"{' '.join(faq.keywords)} " * 2  # Double keywords
            text += f"{' '.join(faq.tags)}"  # Add tags
            corpus.append(text)
        return corpus

    def match(self, query: str, top_k: int = 3) -> List[Tuple[FAQ, float]]:
        """Enhanced matching with multiple strategies."""
        results = []

        # Strategy 1: Full text TF-IDF
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Strategy 2: Question-only TF-IDF
        query_vector_q = self.vectorizer_question.transform([query])
        similarities_q = cosine_similarity(query_vector_q, self.question_matrix).flatten()

        # Combine scores with weights
        for i, faq in enumerate(self.faqs):
            # Weighted combination
            full_score = float(similarities[i]) * 0.6
            question_score = float(similarities_q[i]) * 0.4
            combined_score = full_score + question_score

            # Apply priority boost
            priority_boost = faq.priority / 10.0  # Max 0.5 boost
            final_score = min(combined_score + priority_boost, 1.0)

            results.append((faq, round(final_score, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def best_match(self, query: str, threshold: float = 0.15) -> Optional[Tuple[FAQ, float]]:
        results = self.match(query, top_k=1)
        if results and results[0][1] >= threshold:
            return results[0]
        return None

# ============================================================================
# ENHANCED HYBRID SEARCH
# ============================================================================

class HybridSearch:
    """Enhanced hybrid search with advanced merging strategies."""

    def __init__(self, faqs: List[FAQ]):
        self.faqs = faqs
        self.matcher = FAQMatcher(faqs)
        self.keyword_search = KeywordSearch()

    def search(self, query: str, top_k: int = 3) -> List[Tuple[FAQ, float]]:
        """
        Advanced hybrid search with weighted merging.
        """
        # Get keyword results with scores
        keyword_results = self.keyword_search.search_with_score(query, self.faqs, top_k=len(self.faqs))

        # Get TF-IDF results
        tfidf_results = self.matcher.match(query, top_k=len(self.faqs))

        # Create score mapping with confidence
        merged_scores = {}
        confidence_scores = {}

        # Process keyword results with base score
        for faq, score in keyword_results:
            # Boost keyword matches with base score and dynamic weighting
            keyword_boost = 0.5 + (score * 0.4)  # 0.5-0.9 range
            if faq not in merged_scores or keyword_boost > merged_scores[faq]:
                merged_scores[faq] = round(keyword_boost, 4)

        # Process TF-IDF results
        for faq, score in tfidf_results:
            if faq not in merged_scores or score > merged_scores[faq]:
                merged_scores[faq] = round(score, 4)

        # Apply confidence based on FAQ view count
        for faq in merged_scores:
            # Boost popular FAQs slightly
            view_boost = min(faq.view_count / 1000, 0.1)
            merged_scores[faq] = round(min(merged_scores[faq] + view_boost, 1.0), 4)

        # Sort and return
        results = list(merged_scores.items())
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ============================================================================
# TASK 4: ADVANCED SUPPORT AGENT
# ============================================================================

@dataclass
class ConversationTurn:
    """Enhanced conversation turn with more metadata."""
    role: str
    content: str
    faq_id: str = None
    confidence: float = None
    timestamp: datetime = field(default_factory=datetime.now)
    turn_id: str = field(default_factory=lambda: f"turn-{random.randint(10000, 99999)}")
    sentiment: Optional[str] = None
    processing_time: float = 0.0

class ConversationAnalysis:
    """Analyzes conversation patterns."""

    @staticmethod
    def analyze_sentiment(text: str) -> str:
        """Simple sentiment analysis."""
        positive_words = ["great", "good", "excellent", "thank", "awesome", "helpful", "perfect"]
        negative_words = ["bad", "terrible", "awful", "frustrated", "angry", "disappointed", "useless"]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def get_conversation_stats(turns: List[ConversationTurn]) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        if not turns:
            return {"total_turns": 0}

        user_turns = [t for t in turns if t.role == "user"]
        assistant_turns = [t for t in turns if t.role == "assistant"]

        confidences = [t.confidence for t in assistant_turns if t.confidence is not None]
        avg_confidence = np.mean(confidences) if confidences else 0

        duration = 0
        if len(turns) > 1:
            duration = (turns[-1].timestamp - turns[0].timestamp).total_seconds()

        return {
            "total_turns": len(turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "avg_confidence": avg_confidence,
            "total_faqs_used": len([t for t in assistant_turns if t.faq_id]),
            "duration": duration
        }

class SupportAgent:
    """Advanced helpdesk agent with sophisticated features."""

    def __init__(self, faqs: List[FAQ], llm_client: LLMClient, confidence_threshold: float = 0.15):
        self.faqs = faqs
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.conversation_history: List[ConversationTurn] = []
        self.is_escalated = False
        self.ticket_id = None
        self.low_confidence_streak = 0
        self.max_low_confidence_streak = 3
        self.user_satisfaction: Optional[str] = None

        # Initialize hybrid search
        self.hybrid_search = HybridSearch(faqs)
        self.conversation_analyzer = ConversationAnalysis()

        # Session metadata
        self.session_id = f"session-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        self.start_time = datetime.now()

        logger.info(f"SupportAgent initialized - Session: {self.session_id}")

    def handle_message(self, user_message: str) -> str:
        """
        Handle user message with advanced processing.
        """
        start_time = time.time()

        # Record user message with sentiment
        sentiment = self.conversation_analyzer.analyze_sentiment(user_message)
        user_turn = ConversationTurn(
            role="user",
            content=user_message,
            sentiment=sentiment
        )
        self.conversation_history.append(user_turn)

        logger.info(f"User message received: {user_message[:50]}... (sentiment: {sentiment})")

        # Check if already escalated
        if self.is_escalated:
            response = self._handle_escalated_state()
            self._add_assistant_turn(response, confidence=0.0)
            return response

        # Run hybrid search
        matches = self.hybrid_search.search(user_message, top_k=1)

        if matches:
            best_faq, confidence = matches[0]

            # Check confidence
            if confidence >= self.confidence_threshold:
                # Generate LLM response
                try:
                    response = self.llm_client.generate_faq_response(user_message, best_faq)
                    processing_time = time.time() - start_time
                    self._add_assistant_turn(
                        response,
                        faq_id=best_faq.faq_id,
                        confidence=confidence,
                        processing_time=processing_time
                    )
                    best_faq.increment_views()
                    self.low_confidence_streak = 0

                    logger.info(f"High confidence response generated: {confidence:.2f}, FAQ: {best_faq.faq_id})")
                    return response

                except Exception as e:
                    logger.error(f"LLM generation failed: {str(e)}")
                    # Fallback to FAQ answer
                    fallback = f"I found a relevant FAQ for your question:\n\n{best_faq.answer}\n\n"
                    fallback += "Would you like more details or assistance?"
                    self._add_assistant_turn(
                        fallback,
                        faq_id=best_faq.faq_id,
                        confidence=confidence
                    )
                    self.low_confidence_streak = 0
                    return fallback
            else:
                self.low_confidence_streak += 1
                response = self._generate_fallback_response(user_message)
                processing_time = time.time() - start_time
                self._add_assistant_turn(
                    response,
                    confidence=0.0,
                    processing_time=processing_time
                )
                return response
        else:
            self.low_confidence_streak += 1
            response = self._generate_fallback_response(user_message)
            processing_time = time.time() - start_time
            self._add_assistant_turn(
                response,
                confidence=0.0,
                processing_time=processing_time
            )
            return response

    def _add_assistant_turn(self, content: str, faq_id: str = None,
                          confidence: float = None, processing_time: float = 0.0):
        """Add assistant turn to history."""
        turn = ConversationTurn(
            role="assistant",
            content=content,
            faq_id=faq_id,
            confidence=confidence,
            processing_time=processing_time
        )
        self.conversation_history.append(turn)

    def _generate_fallback_response(self, user_message: str) -> str:
        """Generate intelligent fallback response."""
        response = "I couldn't find a confident match for your question in our knowledge base."

        # Check if we should suggest escalation
        if self.low_confidence_streak >= self.max_low_confidence_streak:
            response += "\n\nI notice this is the third time I couldn't provide a confident answer. "
            response += "Would you like me to escalate this to a human support agent? "
            response += "Just type 'escalate' and I'll create a ticket for you."
        else:
            response += " Would you like to connect with a human support agent? "
            response += "Type 'escalate' to raise a ticket."

        # Add helpful suggestions
        if self.conversation_history:
            suggested_questions = self._generate_suggestions(user_message)
            if suggested_questions:
                response += f"\n\nYou might also try asking:\n• {suggested_questions}"

        return response

    def _generate_suggestions(self, user_message: str) -> Optional[str]:
        """Generate suggested questions based on user message."""
        # Extract key terms
        words = set(user_message.lower().split())
        suggestions = []

        for faq in self.faqs:
            if any(word in faq.question.lower() for word in words):
                suggestions.append(faq.question)
                if len(suggestions) >= 3:
                    break

        return "\n• ".join(suggestions) if suggestions else None

    def _handle_escalated_state(self) -> str:
        """Handle when already escalated."""
        return (f"Your request has already been escalated. "
                f"Please reference ticket #{self.ticket_id} if you need to follow up. "
                "Our team will contact you within 4 business hours.")

    def escalate(self, reason: str = "User requested human support") -> str:
        """
        Escalate to human support with ticket creation.
        """
        if self.is_escalated:
            return self._handle_escalated_state()

        self.is_escalated = True
        self.ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d')}-{random.randint(10000, 99999)}"

        # Get conversation context for the ticket
        context = self.get_conversation_summary()

        response = f"Your request has been escalated to our support team.\n"
        response += f"Ticket ID: {self.ticket_id}\n"
        response += f"Estimated response time: within 4 business hours.\n"
        response += f"Reason: {reason}\n\n"
        response += "A support agent will review your case and reach out to you shortly.\n"
        response += "You can reference this ticket ID in any follow-up communications."

        logger.info(f"Escalation created: {self.ticket_id}")
        self._add_assistant_turn(response, confidence=0.0)
        return response

    def get_conversation_summary(self) -> str:
        """
        Get detailed conversation summary with analytics.
        """
        if not self.conversation_history:
            return "No conversation history available."

        stats = self.conversation_analyzer.get_conversation_stats(self.conversation_history)

        summary = []
        summary.append("="*70)
        summary.append("📋 CONVERSATION SUMMARY")
        summary.append("="*70)
        summary.append(f"Session ID: {self.session_id}")
        summary.append(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"Duration: {stats['duration']:.0f} seconds")
        summary.append(f"Total turns: {stats['total_turns']}")
        summary.append(f"Average confidence: {stats['avg_confidence']:.4f}")
        summary.append(f"FAQs used: {stats['total_faqs_used']}")
        summary.append(f"Escalated: {'Yes' if self.is_escalated else 'No'}")
        if self.is_escalated:
            summary.append(f"Ticket ID: {self.ticket_id}")
        summary.append("="*70)
        summary.append("\n📝 CONVERSATION LOG:")

        for i, turn in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if turn.role == "user" else "🤖"
            role_name = "USER" if turn.role == "user" else "SUPPORTAI"
            summary.append(f"\n[{i}] {role_emoji} {role_name}")
            summary.append(f"    Time: {turn.timestamp.strftime('%H:%M:%S')}")
            if turn.sentiment:
                summary.append(f"    Sentiment: {turn.sentiment}")
            summary.append(f"    Content: {turn.content[:100]}..." if len(turn.content) > 100 else f"    Content: {turn.content}")
            if turn.faq_id:
                summary.append(f"    FAQ: {turn.faq_id} | Confidence: {turn.confidence:.4f}")
            if turn.processing_time > 0:
                summary.append(f"    Processing: {turn.processing_time:.2f}s")
            summary.append("    " + "-"*50)

        summary.append("\n📊 STATISTICS:")
        summary.append(f"• Total user messages: {stats['user_turns']}")
        summary.append(f"• Total assistant responses: {stats['assistant_turns']}")
        summary.append(f"• Average confidence: {stats['avg_confidence']:.4f}")
        summary.append(f"• FAQ matches used: {stats['total_faqs_used']}")

        return "\n".join(summary)

    def reset(self):
        """Reset conversation with analytics logging."""
        logger.info(f"Session {self.session_id} reset")
        self.conversation_history = []
        self.is_escalated = False
        self.ticket_id = None
        self.low_confidence_streak = 0
        self.session_id = f"session-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"
        self.start_time = datetime.now()

# ============================================================================
# ADVANCED CHAT INTERFACE
# ============================================================================

class ChatInterface:
    """Advanced chat interface with rich features."""

    def __init__(self, agent: SupportAgent):
        self.agent = agent
        # Define colors, can be actual ANSI colors or empty strings for compatibility
        self.colors = {
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "red": "\033[91m",
            "reset": "\033[0m",
            "bold": "\033[1m"
        }
        # Disable colors if not in a compatible environment (e.g., Windows cmd without ANSI support)
        # For Colab, ANSI colors usually work, so we keep them.

    def _print_banner(self):
        """Print welcome banner with system status."""
        banner = f"""
{self.colors["bold"]}╔{'═'*60}╗{self.colors["reset"]}
{self.colors["bold"]}║{' ' * 20}SUPPORTAI — HELPDESK AGENT{' ' * 20} ║{self.colors["reset"]}
{self.colors["bold"]}╠{'═'*60}╣{self.colors["reset"]}
{self.colors["bold"]}║ Version: 2.0 | Session: {self.agent.session_id[:20]:<20} ║{self.colors["reset"]}
{self.colors["bold"]}║ Status: {'Active' if not self.agent.is_escalated else 'Escalated':<20} ║{self.colors["reset"]}
{self.colors["bold"]}║ FAQs Available: {len(self.agent.faqs):<17} ║{self.colors["reset"]}
{self.colors["bold"]}║ Confidence Threshold: {self.agent.confidence_threshold:<13.2f} ║{self.colors["reset"]}
{self.colors["bold"]}╚{'═'*60}╝{self.colors["reset"]}
        """
        print(banner)

        commands = f"""
{self.colors["bold"]}{self.colors["blue"]}📋 COMMANDS:{self.colors["reset"]}
  • Type any question to ask SupportAI
  • '{self.colors["yellow"]}history{self.colors["reset"]}' - Show detailed conversation summary
  • '{self.colors["yellow"]}escalate{self.colors["reset"]}' - Escalate to human support
  • '{self.colors["yellow"]}reset{self.colors["reset"]}' - Clear conversation and start fresh
  • '{self.colors["yellow"]}stats{self.colors["reset"]}' - Show system statistics
  • '{self.colors["yellow"]}faqs{self.colors["reset"]}' - List all available FAQs
  • '{self.colors["yellow"]}quit{self.colors["reset"]}' - Exit application
{self.colors["bold"]}─{'─'*59}{self.colors["reset"]}
"""
        print(commands)

    def _print_response(self, response_text: str, confidence: Optional[float], faq_id: Optional[str]):
        """Helper to print assistant responses with metadata."""
        print(f"\n{self.colors['bold']}{self.colors['green']}🤖 SupportAI:{self.colors['reset']}")
        print(f"{response_text}")
        if confidence is not None:
            status_color = self.colors['green'] if confidence >= self.agent.confidence_threshold else self.colors['yellow']
            print(f"{self.colors['bold']}{status_color}Confidence: {confidence:.4f}{self.colors['reset']}", end="")
            if faq_id:
                print(f" | FAQ: {faq_id}", end="")
            if self.agent.conversation_history and self.agent.conversation_history[-1].processing_time > 0:
                print(f" | Processing time: {self.agent.conversation_history[-1].processing_time:.2f}s")
            else:
                print("") # Newline if no processing time printed
        print(f"{self.colors['bold']}─{'─'*59}{self.colors['reset']}")


    def _show_stats(self):
        """Display LLM client and agent statistics."""
        print(f"\n{self.colors['bold']}{self.colors['blue']}📊 SYSTEM STATISTICS:{self.colors['reset']}")
        print(f"{self.colors['bold']}─{'─'*59}{self.colors['reset']}")
        print(f"LLM Client Stats:")
        for key, value in self.agent.llm_client.stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print(f"Cache Size: {len(self.agent.llm_client.cache)} entries")
        print(f"Cache TTL: {self.agent.llm_client.cache_ttl} seconds")
        print(f"{self.colors['bold']}─{'─'*59}{self.colors['reset']}")
        print(f"Agent Stats:")
        conv_stats = self.agent.conversation_analyzer.get_conversation_stats(self.agent.conversation_history)
        for key, value in conv_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"  {key.replace('_', ' ').title()}: {value}")
        print(f"{self.colors['bold']}─{'─'*59}{self.colors['reset']}")

    def _list_faqs(self):
        """List all available FAQs."""
        print(f"\n{self.colors['bold']}{self.colors['blue']}📚 AVAILABLE FAQS:{self.colors['reset']}")
        print(f"{self.colors['bold']}─{'─'*59}{self.colors['reset']}")
        for i, faq in enumerate(self.agent.faqs, 1):
            print(f"{i}. {faq.question} (Category: {faq.category.value})")
        print(f"{self.colors['bold']}─{'─'*59}{self.colors['reset']}")

    def run(self):
        """Run the interactive chat interface."""
        self._print_banner()

        while True:
            try:
                user_input = input(f"\n{self.colors['bold']}You:{self.colors['reset']} ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() == 'quit':
                    print("\nThank you for using SupportAI. Goodbye! 👋")
                    break

                elif user_input.lower() == 'history':
                    print("\n" + self.agent.get_conversation_summary())
                    continue

                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue

                elif user_input.lower() == 'faqs':
                    self._list_faqs()
                    continue

                elif user_input.lower() == 'escalate':
                    response = self.agent.escalate()
                    self._print_response(response, None, None) # Escalation doesn't have a direct FAQ match
                    continue

                elif user_input.lower() == 'reset':
                    self.agent.reset()
                    print(f"\n{self.colors['green']}🔄 Conversation reset. Starting fresh!{self.colors['reset']}")
                    self._print_banner() # Reprint banner after reset
                    continue

                # Process regular message
                response_text = self.agent.handle_message(user_input)

                # Get metadata from the last assistant turn
                last_assistant_turn = None
                for turn in reversed(self.agent.conversation_history):
                    if turn.role == "assistant":
                        last_assistant_turn = turn
                        break

                if last_assistant_turn:
                    self._print_response(
                        last_assistant_turn.content,
                        last_assistant_turn.confidence,
                        last_assistant_turn.faq_id
                    )
                else:
                    # Should not happen if handle_message always adds a turn, but good for robustness
                    self._print_response(response_text, None, None)

            except EOFError:
                print("\nExiting chat due to EOF.")
                break
            except KeyboardInterrupt:
                print("\nExiting chat. Goodbye!")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in chat interface: {e}")
                print(f"{self.colors['red']}An error occurred: {e}{self.colors['reset']}")
                print("Please try again or type 'quit' to exit.")


# ============================================================================
# RUN THE ADVANCED CHAT INTERFACE
# ============================================================================

# --- IMPORTANT: Ensure OPENROUTER_API_KEY is defined in the previous cell ---
# For Google Colab, you might have set it as an environment variable or directly.

# Placeholder for the API key (replace with your actual key or load from env)
# Make sure OPENROUTER_API_KEY is available from a previous cell or your environment
if 'OPENROUTER_API_KEY' not in locals() and 'OPENROUTER_API_KEY' not in globals():
    # Attempt to load from environment variables if not set directly
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "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":
    print("⚠️  WARNING: OPENROUTER_API_KEY is not set or is still the placeholder. "
          "Please set your actual API key to use the LLM functionality.")
    print("You can get an API key from https://openrouter.ai/")
else:
    print("✓ OpenRouter API key detected. Initializing chat interface...")
    # Initialize LLMClient
    llm_client = LLMClient(api_key=OPENROUTER_API_KEY)

    # Initialize SupportAgent with the comprehensive FAQs and LLM client
    support_agent = SupportAgent(faqs=faqs, llm_client=llm_client)

    # Initialize and run the ChatInterface
    chat_interface = ChatInterface(agent=support_agent)
    chat_interface.run()
