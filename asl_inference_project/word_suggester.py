"""
Auto Word Suggestion System for ASL Recognition.

Provides intelligent word suggestions based on:
- Current partial word being typed
- Sentence context
- Common English word frequency
"""

from typing import List, Tuple, Dict
from collections import defaultdict


class WordSuggester:
    """
    Provides intelligent word suggestions for ASL sentence construction.
    """

    def __init__(self):
        """Initialize the word suggester with common English words."""
        # Common English words sorted by frequency
        self.common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "is", "are", "was", "were", "been", "being", "has", "had",
            "did", "does", "do", "should", "would", "could", "may", "might", "must",
            "hello", "hi", "good", "morning", "afternoon", "evening", "night", "yes",
            "no", "please", "thank", "thanks", "you", "welcome", "sorry", "excuse",
            "name", "call", "phone", "number", "address", "email", "help", "need",
            "want", "like", "love", "hate", "know", "understand", "think", "feel",
            "happy", "sad", "angry", "good", "bad", "right", "wrong", "true", "false",
            "today", "tomorrow", "yesterday", "now", "later", "soon", "never", "always",
            "here", "there", "where", "when", "why", "how", "what", "who", "which",
            "that", "this", "these", "those", "my", "your", "his", "her", "its", "our",
            "their", "mine", "yours", "hers", "ours", "theirs", "myself", "yourself",
            "himself", "herself", "itself", "ourselves", "themselves", "each", "every",
            "all", "some", "any", "no", "none", "both", "either", "neither", "one", "two",
            "three", "four", "five", "six", "seven", "eight", "nine", "ten", "first",
            "second", "third", "next", "last", "big", "small", "large", "little", "long",
            "short", "tall", "high", "low", "fast", "slow", "hot", "cold", "warm", "cool",
            "old", "new", "young", "rich", "poor", "expensive", "cheap", "easy", "hard",
            "simple", "complex", "important", "unimportant", "necessary", "unnecessary",
            "possible", "impossible", "likely", "unlikely", "certain", "uncertain"
        ]
        
        # Contextual word pairs for context-aware suggestions
        self.context_pairs = {
            "good": ["morning", "afternoon", "evening", "night", "luck", "job", "bye"],
            "hello": ["there", "world", "everyone", "my", "name", "i"],
            "thank": ["you", "for", "the", "very", "much"],
            "please": ["help", "tell", "show", "give", "let", "do"],
            "how": ["are", "do", "can", "did", "much", "many", "long"],
            "what": ["is", "are", "do", "did", "time", "name", "about"],
            "where": ["is", "are", "do", "did", "to", "from", "at"],
            "when": ["is", "are", "do", "did", "will", "can", "did"],
            "why": ["do", "did", "is", "are", "not", "should"],
            "who": ["is", "are", "do", "did", "will", "can", "are"],
            "i": ["am", "was", "will", "can", "have", "had", "need", "want", "like", "love"],
            "you": ["are", "were", "will", "can", "have", "had", "need", "want", "like", "love"],
            "he": ["is", "was", "will", "can", "has", "had", "needs", "wants", "likes", "loves"],
            "she": ["is", "was", "will", "can", "has", "had", "needs", "wants", "likes", "loves"],
            "we": ["are", "were", "will", "can", "have", "had", "need", "want", "like", "love"],
            "they": ["are", "were", "will", "can", "have", "had", "need", "want", "like", "love"],
            "my": ["name", "friend", "family", "house", "car", "phone", "email", "address"],
            "your": ["name", "friend", "family", "house", "car", "phone", "email", "address"],
            "the": ["first", "last", "best", "worst", "most", "least", "only", "same", "next"],
            "a": ["lot", "few", "little", "bit", "while", "moment", "second", "minute"],
            "to": ["be", "do", "go", "come", "see", "get", "have", "make", "take", "give"],
            "of": ["the", "a", "an", "my", "your", "his", "her", "our", "their"],
            "in": ["the", "a", "an", "my", "your", "order", "fact", "addition", "other"],
            "on": ["the", "a", "an", "my", "your", "top", "bottom", "left", "right"],
            "at": ["the", "a", "an", "my", "your", "home", "work", "school", "office"],
            "by": ["the", "a", "an", "my", "your", "way", "side", "end", "beginning"],
            "for": ["the", "a", "an", "my", "your", "example", "instance", "reason", "purpose"],
            "with": ["the", "a", "an", "my", "your", "me", "you", "him", "her", "us", "them"],
            "from": ["the", "a", "an", "my", "your", "beginning", "start", "end", "top"],
            "about": ["the", "a", "an", "my", "your", "what", "how", "why", "who", "where"],
        }
        
        # User history for personalized suggestions
        self.user_history = defaultdict(int)
        
        print(f"WordSuggester initialized with {len(self.common_words)} common words")

    def get_suggestions(self, prefix: str, context: str = "", max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Get word suggestions based on prefix and context.
        
        Args:
            prefix: Current partial word being typed
            context: Previous word in sentence for context
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of (word, confidence) tuples sorted by relevance
        """
        suggestions = []
        
        # 1. Prefix matching from common words
        prefix_lower = prefix.lower()
        for word in self.common_words:
            if word.startswith(prefix_lower):
                # Higher confidence for exact prefix match
                confidence = 0.9
                suggestions.append((word, confidence))
        
        # 2. Context-aware suggestions
        if context and context.lower() in self.context_pairs:
            context_words = self.context_pairs[context.lower()]
            for word in context_words:
                if word.startswith(prefix_lower):
                    # Boost confidence for context matches
                    confidence = 0.95
                    suggestions.append((word, confidence))
        
        # 3. User history suggestions
        for word, count in self.user_history.items():
            if word.startswith(prefix_lower):
                # Confidence based on usage frequency
                confidence = min(0.85 + (count * 0.01), 0.99)
                suggestions.append((word, confidence))
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for word, conf in suggestions:
            if word not in seen:
                seen.add(word)
                unique_suggestions.append((word, conf))
        
        unique_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return unique_suggestions[:max_suggestions]

    def select_suggestion(self, word: str):
        """
        Record when a user selects a suggestion.
        
        Args:
            word: The word that was selected
        """
        self.user_history[word] += 1

    def reset_history(self):
        """Reset user history."""
        self.user_history.clear()


if __name__ == '__main__':
    # Test the word suggester
    suggester = WordSuggester()
    
    print("Testing word suggestions:")
    print("=" * 50)
    
    # Test prefix matching
    print("Prefix 'hel':", suggester.get_suggestions("hel"))
    print("Prefix 'th':", suggester.get_suggestions("th"))
    
    # Test context-aware
    print("Prefix 'm' with context 'good':", suggester.get_suggestions("m", "good"))
    print("Prefix 'n' with context 'my':", suggester.get_suggestions("n", "my"))
    
    # Test user history
    suggester.select_suggestion("hello")
    suggester.select_suggestion("hello")
    print("Prefix 'hel' after history:", suggester.get_suggestions("hel"))
