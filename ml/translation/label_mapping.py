"""
Label mapping system for ISL translation.

Provides configurable mapping between class IDs and ISL words/phrases with multi-language support.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict


logger = logging.getLogger(__name__)


@dataclass
class LabelEntry:
    """Single label entry with translations."""
    class_id: int
    word: str
    phrase: str = ""
    description: str = ""
    category: str = ""
    
    # Multi-language support
    translations: Dict[str, str] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    
    def get_translation(self, language: str = "en") -> str:
        """
        Get translation for specific language.
        
        Args:
            language: Language code (e.g., 'en', 'hi', 'bn')
            
        Returns:
            Translated word or phrase
        """
        if language == "en" or language not in self.translations:
            return self.phrase if self.phrase else self.word
        return self.translations.get(language, self.word)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LabelEntry':
        """Create LabelEntry from dictionary."""
        return cls(**data)


class LabelMapping:
    """
    Configurable label mapping system for ISL translation.
    
    Independent from ML model - only handles mapping logic.
    """
    
    def __init__(self, default_language: str = "en"):
        """
        Initialize label mapping.
        
        Args:
            default_language: Default language for translations
        """
        self.default_language = default_language
        self.labels: Dict[int, LabelEntry] = {}
        self.reverse_mapping: Dict[str, int] = {}  # word -> class_id
        self.categories: Dict[str, List[int]] = {}  # category -> class_ids
        
        logger.info(f"Initialized LabelMapping with default language: {default_language}")
    
    def add_label(
        self,
        class_id: int,
        word: str,
        phrase: str = "",
        description: str = "",
        category: str = "",
        translations: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ) -> None:
        """
        Add a label entry.
        
        Args:
            class_id: Class ID from model
            word: Primary word for this class
            phrase: Full phrase (optional)
            description: Description of the sign
            category: Category for grouping
            translations: Dictionary of language_code -> translation
            metadata: Additional metadata
        """
        entry = LabelEntry(
            class_id=class_id,
            word=word,
            phrase=phrase,
            description=description,
            category=category,
            translations=translations or {},
            metadata=metadata or {}
        )
        
        self.labels[class_id] = entry
        self.reverse_mapping[word.lower()] = class_id
        
        # Update category mapping
        if category:
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(class_id)
        
        logger.debug(f"Added label: {class_id} -> {word}")
    
    def get_label(self, class_id: int) -> Optional[LabelEntry]:
        """
        Get label entry by class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            LabelEntry or None if not found
        """
        return self.labels.get(class_id)
    
    def get_word(self, class_id: int) -> str:
        """
        Get word for class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            Word string or empty string if not found
        """
        entry = self.get_label(class_id)
        return entry.word if entry else ""
    
    def get_phrase(self, class_id: int, language: str = None) -> str:
        """
        Get phrase for class ID in specified language.
        
        Args:
            class_id: Class ID
            language: Language code (uses default if None)
            
        Returns:
            Phrase string or empty string if not found
        """
        entry = self.get_label(class_id)
        if not entry:
            return ""
        
        lang = language or self.default_language
        return entry.get_translation(lang)
    
    def get_class_id(self, word: str) -> Optional[int]:
        """
        Get class ID from word.
        
        Args:
            word: Word to lookup
            
        Returns:
            Class ID or None if not found
        """
        return self.reverse_mapping.get(word.lower())
    
    def translate(
        self,
        class_id: int,
        language: str = None,
        return_alternatives: bool = False
    ) -> Dict[str, str]:
        """
        Translate class ID to word/phrase.
        
        Args:
            class_id: Class ID to translate
            language: Language code (uses default if None)
            return_alternatives: Whether to return alternative translations
            
        Returns:
            Dictionary with 'word', 'phrase', and optionally 'alternatives'
        """
        entry = self.get_label(class_id)
        if not entry:
            return {'word': '', 'phrase': ''}
        
        lang = language or self.default_language
        
        result = {
            'word': entry.word,
            'phrase': entry.get_translation(lang),
            'description': entry.description,
            'category': entry.category
        }
        
        if return_alternatives:
            result['alternatives'] = [
                entry.translations.get(lang_code, entry.word)
                for lang_code in entry.translations.keys()
                if lang_code != lang
            ]
        
        return result
    
    def get_category_labels(self, category: str) -> List[LabelEntry]:
        """
        Get all labels in a category.
        
        Args:
            category: Category name
            
        Returns:
            List of LabelEntries in the category
        """
        class_ids = self.categories.get(category, [])
        return [self.labels[cid] for cid in class_ids if cid in self.labels]
    
    def get_all_categories(self) -> List[str]:
        """Get all category names."""
        return list(self.categories.keys())
    
    def get_num_classes(self) -> int:
        """Get total number of classes."""
        return len(self.labels)
    
    def get_class_ids(self) -> List[int]:
        """Get all class IDs."""
        return list(self.labels.keys())
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """
        Save label mapping to JSON file.
        
        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'default_language': self.default_language,
            'labels': [entry.to_dict() for entry in self.labels.values()],
            'categories': self.categories
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved label mapping to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'LabelMapping':
        """
        Load label mapping from JSON file.
        
        Args:
            filepath: Path to load file from
            
        Returns:
            LabelMapping instance
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        mapping = cls(default_language=data.get('default_language', 'en'))
        
        for label_data in data.get('labels', []):
            mapping.add_label(**label_data)
        
        logger.info(f"Loaded label mapping from {filepath} with {mapping.get_num_classes()} classes")
        
        return mapping
    
    def validate(self) -> bool:
        """
        Validate label mapping integrity.
        
        Returns:
            True if valid, False otherwise
        """
        # Check for duplicate class IDs
        if len(self.labels) != len(set(self.labels.keys())):
            logger.error("Duplicate class IDs found")
            return False
        
        # Check for missing translations
        for entry in self.labels.values():
            if not entry.word:
                logger.error(f"Empty word for class ID {entry.class_id}")
                return False
        
        # Check reverse mapping consistency
        for word, class_id in self.reverse_mapping.items():
            if class_id not in self.labels:
                logger.error(f"Reverse mapping inconsistency: {word} -> {class_id}")
                return False
        
        logger.info("Label mapping validation passed")
        return True
    
    def create_sample_mapping(self, num_classes: int = 25) -> None:
        """
        Create a sample ISL label mapping for testing.
        
        Args:
            num_classes: Number of sample classes to create
        """
        # Common ISL words/phrases
        sample_labels = [
            ("Hello", "Hello", "Greeting", "greetings"),
            ("Thank you", "Thank you", "Expressing gratitude", "greetings"),
            ("Yes", "Yes", "Affirmation", "common"),
            ("No", "No", "Negation", "common"),
            ("Please", "Please", "Polite request", "common"),
            ("Sorry", "Sorry", "Apology", "greetings"),
            ("Good morning", "Good morning", "Morning greeting", "greetings"),
            ("Good evening", "Good evening", "Evening greeting", "greetings"),
            ("How are you", "How are you", "Asking about well-being", "questions"),
            ("What is your name", "What is your name", "Asking for name", "questions"),
            ("My name is", "My name is", "Introducing oneself", "personal"),
            ("I understand", "I understand", "Showing comprehension", "common"),
            ("I don't understand", "I don't understand", "Showing lack of comprehension", "common"),
            ("Help", "Help", "Requesting assistance", "common"),
            ("Water", "Water", "Requesting water", "needs"),
            ("Food", "Food", "Requesting food", "needs"),
            ("Stop", "Stop", "Command to stop", "commands"),
            ("Go", "Go", "Command to proceed", "commands"),
            ("Come", "Come", "Command to approach", "commands"),
            ("Wait", "Wait", "Command to wait", "commands"),
            ("Look", "Look", "Command to look", "commands"),
            ("Listen", "Listen", "Command to listen", "commands"),
            ("Sit", "Sit", "Command to sit", "commands"),
            ("Stand", "Stand", "Command to stand", "commands"),
            ("Walk", "Walk", "Command to walk", "commands"),
        ]
        
        for i, (word, phrase, description, category) in enumerate(sample_labels[:num_classes]):
            self.add_label(
                class_id=i,
                word=word,
                phrase=phrase,
                description=description,
                category=category,
                translations={
                    'hi': f"{word} (Hindi)",
                    'bn': f"{word} (Bengali)"
                }
            )
        
        logger.info(f"Created sample mapping with {min(num_classes, len(sample_labels))} classes")
