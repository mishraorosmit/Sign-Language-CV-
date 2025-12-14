"""
Text Formation Module

This module handles converting individual letters into words and sentences.
It manages spacing, prevents duplicate letters, and builds coherent text.

BEGINNER NOTE: This is like a smart typewriter that knows when to add spaces
and when to avoid repeating letters.
"""

import time
from config import LETTER_REPEAT_DELAY, SPACE_TRIGGER_DELAY, MAX_SENTENCE_LENGTH


class TextBuilder:
    """
    A class to build text from individual letter predictions.
    
    This handles:
    - Adding letters to form words
    - Preventing duplicate letters from same gesture
    - Adding spaces when appropriate
    - Managing sentence length
    """
    
    def __init__(self):
        """Initialize the text builder."""
        self.current_text = ""
        self.last_letter = None
        self.last_letter_time = 0
        self.last_detection_time = time.time()
        
        print("✓ Text builder initialized")
    
    def add_letter(self, letter):
        """
        Add a letter to the current text.
        
        This prevents adding the same letter multiple times if you hold
        the same gesture.
        
        Args:
            letter: The letter to add (A-Z)
            
        Returns:
            True if letter was added, False if it was a duplicate
        """
        current_time = time.time()
        
        # Check if this is a duplicate letter too soon
        if letter == self.last_letter:
            time_since_last = current_time - self.last_letter_time
            
            if time_since_last < LETTER_REPEAT_DELAY:
                # Too soon - ignore this letter
                return False
        
        # Add the letter
        if len(self.current_text) < MAX_SENTENCE_LENGTH:
            self.current_text += letter
            self.last_letter = letter
            self.last_letter_time = current_time
            self.last_detection_time = current_time
            
            print(f"Added letter: {letter} | Current text: {self.current_text}")
            return True
        else:
            print("⚠ Maximum sentence length reached")
            return False
    
    def check_for_space(self):
        """
        Check if a space should be added.
        
        A space is added if no hand has been detected for SPACE_TRIGGER_DELAY seconds.
        
        Returns:
            True if space was added, False otherwise
        """
        current_time = time.time()
        time_since_detection = current_time - self.last_detection_time
        
        if time_since_detection >= SPACE_TRIGGER_DELAY:
            # Add space if not already at end
            if self.current_text and not self.current_text.endswith(' '):
                if len(self.current_text) < MAX_SENTENCE_LENGTH:
                    self.current_text += ' '
                    print(f"Added space | Current text: {self.current_text}")
                    return True
        
        return False
    
    def update_detection_time(self):
        """
        Update the last detection time.
        
        Call this whenever a hand is detected, even if no letter is added.
        This prevents unwanted spaces.
        """
        self.last_detection_time = time.time()
    
    def delete_last_character(self):
        """
        Delete the last character from the text.
        
        Returns:
            True if character was deleted, False if text was empty
        """
        if self.current_text:
            deleted_char = self.current_text[-1]
            self.current_text = self.current_text[:-1]
            print(f"Deleted: {deleted_char} | Current text: {self.current_text}")
            return True
        return False
    
    def clear_text(self):
        """Clear all text."""
        self.current_text = ""
        self.last_letter = None
        print("✓ Text cleared")
    
    def get_text(self):
        """
        Get the current text.
        
        Returns:
            The current text string
        """
        return self.current_text
    
    def get_word_count(self):
        """
        Get the number of words in the current text.
        
        Returns:
            Number of words
        """
        return len(self.current_text.split())
    
    def get_letter_count(self):
        """
        Get the number of letters (excluding spaces) in the current text.
        
        Returns:
            Number of letters
        """
        return len(self.current_text.replace(' ', ''))


def test_text_builder():
    """
    Test function for the text builder.
    
    This simulates adding letters and demonstrates the text building logic.
    """
    print("\n" + "="*60)
    print("TEXT BUILDER TEST")
    print("="*60)
    
    builder = TextBuilder()
    
    # Test 1: Adding different letters
    print("\nTest 1: Adding different letters")
    builder.add_letter('H')
    time.sleep(0.2)
    builder.add_letter('E')
    time.sleep(0.2)
    builder.add_letter('L')
    time.sleep(0.2)
    builder.add_letter('L')  # This should be blocked (too soon)
    time.sleep(LETTER_REPEAT_DELAY + 0.1)  # Wait long enough
    builder.add_letter('L')  # Now this should work
    time.sleep(0.2)
    builder.add_letter('O')
    
    print(f"\nCurrent text: '{builder.get_text()}'")
    print(f"Letter count: {builder.get_letter_count()}")
    print(f"Word count: {builder.get_word_count()}")
    
    # Test 2: Auto space
    print("\nTest 2: Auto space (waiting for space trigger)")
    print(f"Waiting {SPACE_TRIGGER_DELAY} seconds...")
    time.sleep(SPACE_TRIGGER_DELAY + 0.1)
    builder.check_for_space()
    
    # Test 3: Adding more letters
    print("\nTest 3: Adding more letters")
    builder.update_detection_time()
    builder.add_letter('W')
    time.sleep(0.2)
    builder.add_letter('O')
    time.sleep(0.2)
    builder.add_letter('R')
    time.sleep(0.2)
    builder.add_letter('L')
    time.sleep(0.2)
    builder.add_letter('D')
    
    print(f"\nFinal text: '{builder.get_text()}'")
    print(f"Letter count: {builder.get_letter_count()}")
    print(f"Word count: {builder.get_word_count()}")
    
    # Test 4: Delete
    print("\nTest 4: Delete last character")
    builder.delete_last_character()
    print(f"After delete: '{builder.get_text()}'")
    
    # Test 5: Clear
    print("\nTest 5: Clear all text")
    builder.clear_text()
    print(f"After clear: '{builder.get_text()}'")
    
    print("\n" + "="*60)
    print("✓ Text builder test completed")
    print("="*60)


if __name__ == "__main__":
    test_text_builder()
