# ============================================================================
# TASK 3: Intelligent FAQ Matching
# ============================================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Optional, Dict, Any
import re

# ============================================================================
# TASK 1: FAQ System (Reused)
# ============================================================================

class FAQ:
    def __init__(self, question: str, answer: str, keywords: List[str] = None):
        self.question = question
        self.answer = answer
        self.keywords = keywords or []

# Sample FAQ data (extended with keywords)
faqs = [
    FAQ(
        "How do I reset my password?",
        "To reset your password, go to the login page and click 'Forgot Password'. "
        "Enter the email address associated with your account. You will receive a "
        "password reset link within 5 minutes. The link is valid for 24 hours. "
        "If you don't receive the email, check your spam folder.",
        ["password", "reset", "login", "forgot", "credentials", "account", "email"]
    ),
    FAQ(
        "What are your business hours?",
        "Our support team is available Monday to Friday, 9:00 AM to 6:00 PM EST. "
        "We respond to email inquiries within 2-4 business hours and live chat "
        "requests within 5 minutes during business hours.",
        ["hours", "business", "support", "available", "time", "schedule"]
    ),
    FAQ(
        "How can I update my billing information?",
        "To update your billing information, log into your account and go to "
        "Settings > Billing. You can add a new payment method or update your "
        "billing address there. Changes take effect immediately.",
        ["billing", "payment", "credit card", "update", "change", "address"]
    ),
    FAQ(
        "Do you offer refunds?",
        "Yes, we offer a 30-day money-back guarantee on all annual subscriptions. "
        "Monthly subscriptions can be cancelled at any time but are non-refundable "
        "for the current billing period. Contact support to initiate a refund.",
        ["refund", "money back", "cancel", "subscription", "guarantee", "return"]
    ),
    FAQ(
        "What is your shipping policy?",
        "We offer free standard shipping on all orders over $50. Standard shipping "
        "takes 3-5 business days. Express shipping (2-3 business days) is available "
        "for $15. International shipping takes 7-14 business days.",
        ["shipping", "delivery", "package", "order", "tracking", "delivery time"]
    ),
    FAQ(
        "How do I update my email address?",
        "To update your email address, go to Account Settings > Profile. Click on "
        "the email field, enter your new email, and confirm by clicking 'Save Changes'. "
        "You will receive a verification email to confirm the change.",
        ["email", "update", "change", "profile", "account", "verification"]
    ),
]

def search_by_keyword(query: str, faqs_list: List[FAQ]) -> Optional[FAQ]:
    """
    Search for the best matching FAQ by keyword matching.
    Returns the FAQ with the highest keyword overlap.
    """
    query_words = set(query.lower().split())
    best_match = None
    best_score = 0
    
    for faq in faqs_list:
        # Combine question, answer, and keywords for matching
        text = (faq.question + " " + faq.answer + " " + " ".join(faq.keywords)).lower()
        faq_words = set(text.split())
        overlap = len(query_words.intersection(faq_words))
        if overlap > best_score:
            best_score = overlap
            best_match = faq
    
    return best_match if best_score > 0 else None

def search_by_keyword_with_score(query: str, faqs_list: List[FAQ]) -> List[Tuple[FAQ, float]]:
    """
    Search for matching FAQs by keyword and return with scores.
    Returns list of (FAQ, score) tuples sorted by score descending.
    """
    query_words = set(query.lower().split())
    results = []
    
    for faq in faqs_list:
        text = (faq.question + " " + faq.answer + " " + " ".join(faq.keywords)).lower()
        faq_words = set(text.split())
        overlap = len(query_words.intersection(faq_words))
        if overlap > 0:
            # Normalize score to 0-1 range (max possible overlap is length of query)
            max_possible = len(query_words)
            score = overlap / max_possible if max_possible > 0 else 0
            results.append((faq, round(score, 4)))
    
    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ============================================================================
# TASK 3: FAQMatcher Class with TF-IDF
# ============================================================================

class FAQMatcher:
    """
    Intelligent FAQ matching using TF-IDF vectorization and cosine similarity.
    """
    
    def __init__(self, faqs: List[FAQ]):
        """
        Initialize the FAQMatcher with a list of FAQs.
        Builds a TF-IDF index over each FAQ's question and keywords.
        
        Args:
            faqs: List of FAQ objects
        """
        self.faqs = faqs
        
        # Build corpus by combining question and keywords
        self.corpus = [
            f"{faq.question} {' '.join(faq.keywords)}" 
            for faq in faqs
        ]
        
        # Initialize and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams for better matching
            max_df=0.85,  # Ignore terms that appear in too many documents
            min_df=1
        )
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        
    def match(self, query: str, top_k: int = 3) -> List[Tuple[FAQ, float]]:
        """
        Match a query against FAQs using TF-IDF and cosine similarity.
        
        Args:
            query: The user's question
            top_k: Number of top results to return
            
        Returns:
            List of (faq, confidence_score) tuples, sorted by score descending
            Scores are floats between 0.0 and 1.0, rounded to 4 decimal places
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity between query and all FAQs
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Create list of (faq, score) tuples
        results = []
        for i, faq in enumerate(self.faqs):
            score = float(similarities[i])
            # Round to 4 decimal places
            score = round(score, 4)
            results.append((faq, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def best_match(self, query: str, threshold: float = 0.15) -> Optional[Tuple[FAQ, float]]:
        """
        Return the single best match if its score meets or exceeds the threshold.
        
        Args:
            query: The user's question
            threshold: Minimum confidence score (0.0 to 1.0)
            
        Returns:
            (faq, confidence_score) tuple if best match meets threshold, else None
        """
        results = self.match(query, top_k=1)
        
        if results and results[0][1] >= threshold:
            return results[0]
        return None
    
    def explain_match(self, query: str) -> str:
        """
        Return a formatted string showing the top 3 matches with their scores.
        
        Args:
            query: The user's question
            
        Returns:
            Formatted string with match results
        """
        results = self.match(query, top_k=3)
        
        if not results:
            return "No matches found."
        
        output = []
        for i, (faq, score) in enumerate(results, 1):
            output.append(f"  {i}. [{score:.4f}] {faq.question}")
        
        return "\n".join(output)

# ============================================================================
# HYBRID SEARCH (Combines Keyword and TF-IDF)
# ============================================================================

def hybrid_search(faqs: List[FAQ], query: str, top_k: int = 3) -> List[Tuple[FAQ, float]]:
    """
    Combines keyword search and TF-IDF matching for better results.
    
    Args:
        faqs: List of FAQ objects
        query: The user's question
        top_k: Number of top results to return
        
    Returns:
        List of (faq, confidence_score) tuples, sorted by score descending
    """
    # Get keyword search results with scores
    keyword_results = search_by_keyword_with_score(query, faqs)
    
    # Get TF-IDF results
    matcher = FAQMatcher(faqs)
    tfidf_results = matcher.match(query, top_k=len(faqs))
    
    # Merge results, keeping highest score per FAQ
    merged_scores = {}
    
    # Process keyword results with base score of 0.5
    for faq, score in keyword_results:
        # Boost keyword matches with base score of 0.5
        boosted_score = 0.5 + (score * 0.5)  # Range: 0.5 to 1.0
        if faq not in merged_scores or boosted_score > merged_scores[faq]:
            merged_scores[faq] = round(boosted_score, 4)
    
    # Process TF-IDF results (already 0-1 range)
    for faq, score in tfidf_results:
        if faq not in merged_scores or score > merged_scores[faq]:
            merged_scores[faq] = round(score, 4)
    
    # Convert to list and sort by score descending
    merged_results = list(merged_scores.items())
    merged_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k results
    return merged_results[:top_k]

# ============================================================================
# TASK 4: Demonstration and Comparison
# ============================================================================

def demonstrate_comparison():
    """
    Demonstrate the difference between keyword search, TF-IDF matching,
    and hybrid search with test queries.
    """
    
    print("="*80)
    print("📊 TASK 3: INTELLIGENT FAQ MATCHING - COMPARISON DEMONSTRATION")
    print("="*80)
    
    # Initialize FAQMatcher
    matcher = FAQMatcher(faqs)
    
    # Test queries
    test_queries = [
        ("I forgot my login credentials", "Password reset FAQ"),
        ("Can I get my money back?", "Refund policy FAQ"),
        ("package delivery time", "Shipping FAQ")
    ]
    
    for query, expected in test_queries:
        print(f"\n{'─'*80}")
        print(f"🔍 Query: {query}")
        print(f"Expected Best Match: {expected}")
        print(f"{'─'*80}\n")
        
        # 1. Keyword Search
        print("[Keyword Search]")
        keyword_results = search_by_keyword_with_score(query, faqs)
        if keyword_results:
            for i, (faq, score) in enumerate(keyword_results[:3], 1):
                print(f"  {i}. [{score:.4f}] {faq.question}")
        else:
            print("  (no results)")
        
        # 2. TF-IDF Matching
        print("\n[TF-IDF Matching]")
        tfidf_results = matcher.match(query, top_k=3)
        if tfidf_results:
            for i, (faq, score) in enumerate(tfidf_results, 1):
                print(f"  {i}. [{score:.4f}] {faq.question}")
        else:
            print("  (no results)")
        
        # 3. Hybrid Search
        print("\n[Hybrid Search]")
        hybrid_results = hybrid_search(faqs, query, top_k=3)
        if hybrid_results:
            for i, (faq, score) in enumerate(hybrid_results, 1):
                print(f"  {i}. [{score:.4f}] {faq.question}")
        else:
            print("  (no results)")
        
        # Best match using TF-IDF with threshold
        best = matcher.best_match(query, threshold=0.15)
        if best:
            faq, score = best
            print(f"\n✅ Best match: {faq.question} (confidence: {score:.4f})")
        else:
            print("\n❌ No match found above threshold")
        
        print("\n")

    # =========================================================================
    # ADDITIONAL ANALYSIS: Show why TF-IDF improves on keyword search
    # =========================================================================
    
    print("="*80)
    print("📈 ANALYSIS: WHY TF-IDF IMPROVES ON KEYWORD SEARCH")
    print("="*80)
    
    # Detailed example showing the improvement
    query = "I forgot my login credentials"
    print(f"\nQuery: '{query}'")
    print("\n1. Keyword Search issues:")
    print("   - Exact word matching only")
    print("   - Misses synonyms (forgot -> reset, credentials -> password)")
    print("   - No semantic understanding")
    
    print("\n2. TF-IDF advantages:")
    print("   - Captures semantic similarity")
    print("   - Handles synonyms naturally")
    print("   - Considers word importance across documents")
    print("   - Uses n-grams for phrase matching")
    
    print("\n3. Hybrid Search benefits:")
    print("   - Combines exact keyword matching (base score 0.5)")
    print("   - Adds semantic TF-IDF matching")
    print("   - Gives higher weight to direct keyword hits")
    print("   - More robust for varied query phrasing")
    
    # Show feature extraction example
    print("\n4. Feature Extraction Example:")
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    sample_corpus = [
        "How do I reset my password?",
        "I forgot my login credentials"
    ]
    tfidf = vectorizer.fit_transform(sample_corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"   Top TF-IDF features for 'password reset':")
    for doc_idx in range(2):
        scores = tfidf[doc_idx].toarray().flatten()
        top_indices = scores.argsort()[-5:][::-1]
        print(f"   {'FAQ' if doc_idx == 0 else 'Query'}:")
        for idx in top_indices:
            if scores[idx] > 0:
                print(f"     - '{feature_names[idx]}': {scores[idx]:.4f}")

# ============================================================================
# EXTRA: Helper function to visualize the matching process
# ============================================================================

def visualize_matching(query: str):
    """
    Visualize the matching process for a given query.
    Shows how different methods rank the FAQs.
    """
    matcher = FAQMatcher(faqs)
    
    print(f"\n{'='*80}")
    print(f"🔍 VISUALIZING MATCHING FOR: '{query}'")
    print(f"{'='*80}\n")
    
    # Get all results
    keyword_results = search_by_keyword_with_score(query, faqs)
    tfidf_results = matcher.match(query, top_k=len(faqs))
    hybrid_results = hybrid_search(faqs, query, top_k=len(faqs))
    
    # Create comparison table
    print(f"{'FAQ':<40} {'Keyword':<10} {'TF-IDF':<10} {'Hybrid':<10}")
    print("-" * 70)
    
    # Get all unique FAQs
    all_faqs = set()
    for faq, _ in keyword_results:
        all_faqs.add(faq)
    for faq, _ in tfidf_results:
        all_faqs.add(faq)
    
    # Create score dictionaries
    keyword_scores = {faq: score for faq, score in keyword_results}
    tfidf_scores = {faq: score for faq, score in tfidf_results}
    hybrid_scores = {faq: score for faq, score in hybrid_results}
    
    # Display scores
    for faq in sorted(all_faqs, key=lambda x: hybrid_scores.get(x, 0), reverse=True):
        name = faq.question[:37] + "..." if len(faq.question) > 40 else faq.question
        kw_score = keyword_scores.get(faq, 0.0000)
        tf_score = tfidf_scores.get(faq, 0.0000)
        hy_score = hybrid_scores.get(faq, 0.0000)
        print(f"{name:<40} {kw_score:<10.4f} {tf_score:<10.4f} {hy_score:<10.4f}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 TASK 3: INTELLIGENT FAQ MATCHING")
    print("="*80)
    print("\nThis demonstrates:\n")
    print("1. FAQMatcher class with TF-IDF vectorization")
    print("2. match() with cosine similarity scoring")
    print("3. best_match() with threshold logic")
    print("4. hybrid_search() combining keyword and TF-IDF")
    print("5. Comparison demonstration with 3 test queries")
    print("6. Visual analysis of matching methods\n")
    
    # Run the main demonstration
    demonstrate_comparison()
    
    # Optional: Visualize matching for a specific query
    print("\n" + "="*80)
    print("📊 DETAILED SCORE COMPARISON")
    print("="*80)
    visualize_matching("I forgot my login credentials")
    
    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE")
    print("="*80)
    
    print("\n📋 KEY TAKEAWAYS:")
    print("✓ TF-IDF captures semantic meaning beyond exact keywords")
    print("✓ Cosine similarity provides continuous confidence scores")
    print("✓ Threshold filtering prevents false positive matches")
    print("✓ Hybrid search combines strengths of both methods")
    print("✓ Better handling of paraphrased and reworded questions")