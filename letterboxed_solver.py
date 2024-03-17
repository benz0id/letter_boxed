import string
from dataclasses import dataclass
from typing import Dict, List, Union, Set, Tuple, Iterable
from bs4 import BeautifulSoup
from pathlib import Path
from itertools import product
import random

@dataclass
class WordAttrs:
    """
    Represents the attributes of a word, including its first and last letters,
    the set of unique letters it contains, and the word itself.

    :param word: The word as a string.
    """

    first: str
    last: str
    letters: frozenset
    word: str

    def __init__(self, word: str) -> None:
        """
        Initialize the WordAttrs object with the given word.

        :param word: The word to analyze and store attributes of.
        """
        word = word.lower()
        self.word = word
        self.first = word[0]
        self.last = word[-1]
        self.letters = frozenset([letter for letter in word])


class FLWordDict:
    """
    A dictionary to efficiently find words by their first and last letters,
    and the unique set of letters they contain.

    Optimally selects words for each first and last letter pair, where words
    that contain more unique letters are considered better.
    """

    # Find words via [<first letter>][<last letter>][<set of unique letters>]
    d: Dict[str, Dict[str, Dict[frozenset, WordAttrs]]]
    entries: Dict[str, WordAttrs]

    def __init__(self):
        """
        Initialize an empty dictionary to store words organized by their first
        and last letters, and the unique sets of letters they contain.
        """
        self.d: Dict[str, Dict[str, Dict[frozenset, WordAttrs]]] = {}
        self.entries: Dict[str, WordAttrs] = {}

        for l1 in string.ascii_lowercase:
            self.d[l1] = {}
            for l2 in string.ascii_lowercase:
                self.d[l1][l2] = {}

    def subset(self, first: str, last: str) -> None:
        """
        Reduces the dictionary to contain only words with the specified first
        and last letters.

        :param first: The first letter to filter by.
        :param last: The last letter to filter by.
        """
        for letter in string.ascii_lowercase:
            if letter != first:
                del self.d[letter]

        for letter in string.ascii_lowercase:
            if letter != last:
                del self.d[first][letter]

    def remove_word(self, word: str) -> None:
        """
        Removes a word from the dictionary.

        :param word: The word to remove.
        """
        word_attrs = WordAttrs(word)
        del self.d[word_attrs.first][word_attrs.last][word_attrs.letters]
        del self.entries[word_attrs.word]

    def add_word(self, word: str) -> bool:
        """
        Adds a word to the dictionary.

        :param word: The word to add.
        :return: True if the word was added, False otherwise.
        """
        word_attrs = WordAttrs(word)
        return self.add_wordattrs(word_attrs)

    def add_wordattrs(self, new_word_attrs: WordAttrs) -> bool:
        """
        Adds a WordAttrs object to the dictionary if it is deemed useful,
        removes WordAttrs from the dictionary that are made obsolete by the
        addition of this word.

        This word will be added to the set of words with the same first
        and last letter pair iff the set of its characters are not a subset of
        a word already in the dictionary.

        If two words have the same set of characters, precedence is given to
        the longer word.

        :param new_word_attrs: A WordAttrs object to add.
        :return: True if the WordAttrs was added, False otherwise.
        """
        current_entries = self.d[new_word_attrs.first][new_word_attrs.last]
        new_word_letter_set = new_word_attrs.letters
        known_word_letter_sets = current_entries.keys()

        to_del = []

        for known_word_letter_set in known_word_letter_sets:
            known_word_attrs = current_entries[known_word_letter_set]

            # Descriptors for new word.
            lesser_length = len(known_word_attrs.word) < len(new_word_attrs.word)
            equal_letter_content = known_word_letter_set == new_word_letter_set

            if equal_letter_content and not lesser_length:
                return False

            is_subset = known_word_letter_set.issuperset(new_word_letter_set)

            if is_subset and not equal_letter_content:
                return False

            elif is_subset and equal_letter_content and lesser_length:
                to_del.append(known_word_attrs)
                continue

            is_superset = known_word_letter_set.issubset(new_word_letter_set)
            if is_superset:
                to_del.append(known_word_attrs)

        for word_attrs in to_del:
            self.remove_word(word_attrs.word)

        current_entries[new_word_attrs.letters] = new_word_attrs
        self.entries[new_word_attrs.word] = new_word_attrs

        return True

    def __repr__(self) -> str:
        """
        Provides a string representation of the FLWordDict object.

        :return: A string representation of the dictionary.
        """
        s = ''

        for first_char in string.ascii_lowercase:
            first_lines = ''

            for last_char in string.ascii_lowercase:
                last_lines = ''

                letter_sets = sorted(self.d[first_char][last_char].keys())

                for letter_set in letter_sets:
                    word = self.d[first_char][last_char][letter_set].word
                    last_lines += '\t\t' + word + '\t' + str(
                        set(letter_set)) + '\n'

                if last_lines:
                    first_lines += f'\t{last_char}\n{last_lines}'
            if first_lines:
                s += f'{first_char}\n{first_lines}'

        return s


def has_repeating_chars(word: str) -> bool:
    """
    Determines if the given word has consecutive repeating characters.

    :param word: The word to check.
    :return: True if the word has repeating characters, False otherwise.
    """
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            return True
    return False


class Board:
    """
    Represents a game board, storing information about the letters
    present and their arrangements.
    """

    def __init__(self, rows: Tuple[str, str, str, str]) -> None:
        """
        Initializes the Board with rows of letters.

        :param rows: A tuple of strings, each representing a row of letters on
        the board.
        """
        self.letters: Set[str] = set()
        self.sets: List[Set[str]] = []

        for row in rows:
            letters = set([char for char in row])
            self.sets.append(letters)
            self.letters = self.letters.union(letters)

    def get_set(self, letter: str) -> Union[int, None]:
        """
        Finds the set (row) index that contains the given letter.

        :param letter: The letter to find.
        :return: The index of the set containing the letter, or None if not found.
        """
        for i, letters in enumerate(self.sets):
            if letter in letters:
                return i
        return None

    def is_valid_letter(self, letter: str) -> bool:
        """
        Checks if the given letter is present on the board.

        :param letter: The letter to check.
        :return: True if the letter is present, False otherwise.
        """
        return letter in self.letters

    def is_valid_word(self, word: str) -> bool:
        """
        Determines if a word can be constructed from the letters on the board
        without reusing a set of letters.

        :param word: The word to check.
        :return: True if the word is valid, False otherwise.
        """
        prev_set = -1
        for letter in word.lower():
            if not self.is_valid_letter(letter):
                return False

            set_index = self.get_set(letter)
            if set_index == prev_set:
                return False
            prev_set = set_index
        return True


def compile_dictionary() -> None:
    """
    Compiles a list of words from the GNU Collaborative International
    Dictionary of English and saves them in a text file.

    Removes proper nouns, obsolete words, and malformed entries.
    """
    outfile = Path('cleaned_words.txt')
    if not outfile.exists():
        base_path = 'gcide-0.53/CIDE.'
        words = []
        num_words_parsed = 0

        for letter in string.ascii_uppercase:
            alpha_path = base_path + letter
            # There are 3 bad characters across all files - no major loses when
            # using errors='replace'.
            with open(alpha_path, 'r', errors='replace') as infile:

                soup = BeautifulSoup(infile, 'html.parser')

                for p in soup.find_all('p'):
                    entry = {
                        'title': p.find('ent').text if p.find('ent') else None,
                        'type': p.find('pos').text if p.find('pos') else None,
                        'mark': p.find('mark').text if p.find('mark') else None,
                    }
                    words.append((entry['title'], entry['type'], entry['mark']))
                    num_words_parsed += 1

                    if num_words_parsed % 1000 == 0:
                        print(f'{num_words_parsed}\t-\t{alpha_path}')

        with open(outfile, 'w') as f:
            for word, word_type, mark in words:

                if not isinstance(word, str):
                    continue

                if word_type is not None and 'prop' in word_type:
                    continue

                # Remove words marked as obsolete.
                if mark is not None and 'Obs.' in mark:
                    continue

                f.write(word + '\n')


def get_allowable_words(board: Board,
                        words: Union[List[Tuple[str, str, str]], List[str]],
                        verbose: bool = False) -> FLWordDict:
    """
    Selects optimal words from a list based on criteria such as no repeating characters,
    validity on the given board, and not being obsolete or a proper noun.

    :param board: The game board to validate words against.
    :param words: A list of words or tuples containing the word and its attributes.
    :return: An FLWordDict object containing the optimal words.
    """

    optimal_words = FLWordDict()
    num_bad_entries = 0
    num_not_alpha = 0
    board_invalidated = 0
    added = 0

    for word in words:

        if not isinstance(word, str):
            num_bad_entries += 1
            continue

        if not word.isalpha():
            num_not_alpha += 1
            continue

        if not board.is_valid_word(word):
            board_invalidated += 1
            continue

        added += 1
        optimal_words.add_word(word)
    if verbose:
        print(f'\n Total Entries: {len(words)}'
              f'\n Bad Entries: {num_bad_entries}'
              f'\n Non-Alphabetical Entries: {num_not_alpha}'
              f'\n Incompatible with Board: {board_invalidated}'
              f'\n Considered: {added}'
              f'\n Final Optimal Words: {len(optimal_words.entries)}')
    return optimal_words


class LetterChainIterator:
    """
    Iterates through all possible chains of letters that could link words
    together. Starts with a chain of length 2, increasing to greater as it
    exhausts the possibilities for each length of chain.

    EG:
    [B - A - C - S]
    Could be a chain for
    Banana - Acrylic - Cactus

    Honest in retrospect much more of this could be done using itertools.
    """
    # Current iteration.
    i: int

    # The characters remaining at each position in the chain.
    remaining_options: List[List[str]]

    # The chain itself.
    chain: List[str]

    stochastic: bool = False

    def __init__(self, stochastic: bool = False) -> None:
        """
        Initializes the iterator.
        :param stochastic: If True, the chains are generated in a random order,
        otherwise iteration is done alphabetically.
        """
        self.stochastic = stochastic
        self.i = 0
        self.letters = [letter for letter in string.ascii_lowercase]
        self.remaining_options = [self.letters.copy(), self.letters.copy()]
        self.chain = [self.get_next_state(0), self.get_next_state(1)]

    def get_next_state(self, pos: int) -> str:
        """
        For a given <pos> in the chain, choose the next value from the list of
        remaining allowable values.

        :param pos: The level for which to get the next state.
        :return: The next letter for the specified level.
        """
        if self.stochastic:
            return self.remaining_options[pos].pop(
                random.randint(0, len(self.remaining_options[pos]) - 1))
        else:
            return self.remaining_options[pos].pop(0)

    def __next__(self) -> List[str]:
        """
        Advances the iterator to the next state, generating a new letter chain.

        This could have been done in like 4 lines with itertools...

        :return: A list of strings representing the current state of the letter chain.
        """
        # Skip first iteration.
        if self.i == 0:
            self.i += 1
            return self.chain
        self.i += 1

        pos = len(self.remaining_options) - 1

        can_term = False
        while not can_term:
            empty_layer = len(self.remaining_options[pos]) == 0

            # Flip the state of the <pos> in the chain.
            if not empty_layer:
                self.chain[pos] = self.get_next_state(pos)
                can_term = True

            # We have exhausted the possibilities at position in the chain.
            # Reset this position and iterate the previous one.
            elif empty_layer and pos > -1:
                self.remaining_options[pos] = self.letters.copy()
                self.chain[pos] = self.get_next_state(pos)
                pos -= 1

            # We have exhausted all possibilities for this chain length.
            # Increment chain length and reset.
            if empty_layer and pos == 0:
                self.remaining_options = [self.letters.copy() for _ in
                                          range(len(self.remaining_options) + 1)]
                self.chain = []
                for pos in range(len(self.remaining_options)):
                    self.chain.append(self.get_next_state(pos))
                can_term = True
        return self.chain

    def get_word_attr_iterator(self, word_dict: FLWordDict) \
            -> Union[Iterable[List[WordAttrs]], None]:
        """
        Get an iterator over all possible sets of words in <word_dict> that
        match the current chain.

        :param word_dict: An FLWordDict object containing some number of
            optimised words.
        :return: An iterator over lists of WordAttrs objects, or None if
            <word_dict> cannot satisfy the current chain.
        """
        possible_solutions = []
        for i in range(len(self.chain) - 1):
            first = self.chain[i]
            last = self.chain[i + 1]
            possible_words = list(word_dict.d[first][last].values())
            if len(possible_words) == 0:
                return None
            possible_solutions.append(possible_words)
        return product(*possible_solutions)


def find_solutions(board: Board,
                        optimal_words_dict: FLWordDict,
                        max_depth: int,
                        num_solutions_to_find: int = 10,
                        max_iterations: int = 1000000,
                        stochastic: bool = True,
                        print_every: int = 10000,
                        verbose: bool = False) -> List[List[WordAttrs]]:
    """
    Constructs solutions for a word puzzle given a board and a dictionary
    of optimized words, adhering to constraints such as maximum depth and the
    number of solutions to find.

    :param board: A Board object representing the game board, with methods
        to check the validity of words.
    :param optimal_words_dict: An instance of FLWordDict containing optimal
        words to be used in constructing solutions.
    :param max_depth: The maximum depth to explore in the solution space.
    :param num_solutions_to_find: The target number of solutions to find before
        stopping the search. Default is 10.
    :param max_iterations: The maximum number of iterations to perform in
        the search for solutions. Default is 1,000,000.
    :param stochastic: A boolean indicating whether the search should be
        stochastic (random) or not. Default is True.
    :param print_every: The frequency (in number of iterations) at which
        progress information should be printed. Default is 10,000.
    :return: A list of lists, where each inner list represents a solution,
        containing WordAttrs objects.
    """
    solutions = []
    iterator = LetterChainIterator(stochastic)
    i = 0

    best_set = set()

    while True:
        iterator.__next__()

        if len(iterator.chain) > max_depth:
            return solutions

        # Attempt to get possible solutions based on the current iterator state
        possible_solutions = iterator.get_word_attr_iterator(optimal_words_dict)

        if possible_solutions is None:
            continue

        # Iterate over each possible solution
        for possible_solution in possible_solutions:
            all_letters = set()
            i += 1

            # Accumulate letters from all words in the possible solution
            for word in possible_solution:
                all_letters = all_letters.union(word.letters)

            if all_letters.issuperset(board.letters):
                solutions.append(possible_solution)

            if len(all_letters.intersection(board.letters)) > len(best_set):
                best_set = all_letters.intersection(board.letters)

            if len(solutions) >= num_solutions_to_find:
                return solutions

            if i >= max_iterations:
                return solutions

            # Periodically print progress information
            if i % print_every == 0 and verbose:
                print(f'{i}'
                      f'\n\t Best Set: {best_set}'
                      f'\n\t Depth   : {len(iterator.chain)}')
