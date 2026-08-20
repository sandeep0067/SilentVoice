"""
Test script for word suggestion system.
"""

from word_suggester import WordSuggester

def test_word_suggestions():
    """Test the word suggestion system."""
    print("Testing Word Suggestion System")
    print("=" * 50)
    
    suggester = WordSuggester()
    
    # Test 1: Prefix matching
    print("\nTest 1: Prefix matching")
    print("Prefix 'hel':", suggester.get_suggestions("hel"))
    print("Prefix 'th':", suggester.get_suggestions("th"))
    print("Prefix 'go':", suggester.get_suggestions("go"))
    
    # Test 2: Context-aware suggestions
    print("\nTest 2: Context-aware suggestions")
    print("Prefix 'm' with context 'good':", suggester.get_suggestions("m", "good"))
    print("Prefix 'n' with context 'my':", suggester.get_suggestions("n", "my"))
    print("Prefix 'y' with context 'thank':", suggester.get_suggestions("y", "thank"))
    
    # Test 3: User history
    print("\nTest 3: User history learning")
    suggester.select_suggestion("hello")
    suggester.select_suggestion("hello")
    suggester.select_suggestion("hello")
    print("After selecting 'hello' 3 times:")
    print("Prefix 'hel':", suggester.get_suggestions("hel"))
    
    # Test 4: Empty prefix
    print("\nTest 4: Empty prefix")
    print("Prefix '':", suggester.get_suggestions(""))
    
    # Test 5: No matches
    print("\nTest 5: No matches")
    print("Prefix 'xyz':", suggester.get_suggestions("xyz"))
    
    print("\n" + "=" * 50)
    print("Word suggestion system test completed!")

if __name__ == '__main__':
    test_word_suggestions()
