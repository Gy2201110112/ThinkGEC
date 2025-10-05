import re
from typing import Dict, Tuple, Optional
import spacy
from verl.utils.reward_score.m2scorer import cal_m2scorer
from langdetect import detect

german_pattern = re.compile(r'^[A-Za-zäöüÄÖÜß0-9 ,.!?;:\-$$$$\'\"\\n\\t]*$')
nlp = spacy.load("de_core_news_sm")

def is_all_german_or_punctuation(s):
    return bool(german_pattern.match(s))

def is_german_word(word):
    """简单判断一个单词是否可能是德语"""
    doc = nlp(word)
    if len(doc) == 0:
        return False
    token = doc[0]
    # 排除纯符号和数字
    if token.is_punct or token.is_digit or token.like_num:
        return False
    # 德语词性通常会被正确标注
    return token.pos_ in {"NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "AUX"}


def is_majority_german_words(sentence, threshold=0.3):
    sentence = sentence.strip()
    if not sentence:
        return False

    # 先检测整体语言是否为德语
    try:
        lang = detect(sentence)
        if lang != 'de':
            return False
    except Exception as e:
        print(f"Language detection error: {e}")
        return False

    doc = nlp(sentence)
    words = [token.text for token in doc if not (token.is_space or token.is_punct)]

    german_count = 0
    total_count = len(words)

    if total_count == 0:
        return False

    for word in words:
        if is_german_word(word):
            german_count += 1

    german_ratio = german_count / total_count

    print(f"German word ratio: {german_ratio:.2f} ({german_count}/{total_count})")
    return german_ratio >= threshold


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    # Extract final answer using XML-style tags
    # answer_pattern = r'<answer>(.*?)</answer>'
    answer_pattern = r'</think>(.*)'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if not matches:
        print("[Error] No valid answer tags found")
        return None, solution_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, solution_str


def parse_solution_text_format(solution_text: str) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.

    Args:
        solution_text: Formatted solution text from dataset

    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")

    return status_dict


def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b',
            re.IGNORECASE
        )
        match = pattern.search(answer_text)

        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None

    return status_dict


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    # tags = {
    #     'think_start': ('<think>', 1),
    #     'think_end': ('</think>', 1),
    #     'answer_start': ('<answer>', 1),
    #     'answer_end': ('</answer>', 1)
    # }

    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order

    # if (positions['think_start'] > positions['think_end'] or
    #         positions['think_end'] > positions['answer_start'] or
    #         positions['answer_start'] > positions['answer_end']):
    #     print("  [Error] Incorrect tag order: Expected <step>...</step><answer>...</answer>")
    #     validation_passed = False
    # else:
    #     print("  Tag sequence validation passed")
    if (positions['think_start'] > positions['think_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def split_str_into_sentences(article):
    sentences = []
    current_sentence = ''
    in_quotes = False
    quote_chars = ['"', "'", '“', '”', '‘', '’', '`', '「', '」']
    delimiters = ['.', '!', '?', '。', '？', '！']
    abbreviations = {'Mr', 'Mrs', 'Dr', 'Prof', 'St', 'Co', 'Mt', 'vs', 'etc'}  # 缩写词集合

    i = 0
    while i < len(article):
        char = article[i]
        current_sentence += char

        # 判断是否进入/退出引号
        if char in quote_chars:
            in_quotes = not in_quotes

        # 如果当前字符是可能的断句符
        if char in delimiters and not in_quotes:
            # 查看下一个字符是否为空格或结尾
            next_char_index = i + 1
            if next_char_index < len(article) and not article[next_char_index].isspace():
                # 如果不是空格，再进一步判断是否是缩写
                # 比如检查前面是否有类似 Mr. 这样的结构
                word_before = ''
                j = i - 1
                while j >= 0 and (article[j].isalpha() or article[j] == '.'):
                    word_before = article[j] + word_before
                    j -= 1
                # 去掉末尾的标点
                word_before = word_before.rstrip(''.join(delimiters))
                if word_before in abbreviations:
                    i += 1
                    continue  # 不断句

            # 如果是句子结尾，且不是缩写，就断句
            stripped = current_sentence.strip()
            if stripped:
                sentences.append(stripped)
            current_sentence = ''

        i += 1

    # 添加最后剩下的内容
    stripped = current_sentence.strip()
    if stripped:
        sentences.append(stripped)

    return sentences

def split_sentences_into_word(sentence):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    return tokens

def compute_score(solution_str: str,
                  ground_truth: str,
                  format_reward: int = 1,
                  answer_reward: float = 1.0):
    """Computes comprehensive score for model response.

    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness

    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "=" * 80)
    print(" Processing New Sample ".center(80, '='))

    # Parse ground truth data

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    # answer_text = solution_str
    processed_str = solution_str
    print(f"\n[Model Response]\n{processed_str}")

    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    # print(f"  Format score: {format_score}")
    ground_truth = ground_truth.split("\n")
    # Validate answer content
    if format_correct and answer_text and len(answer_text) <= 2 * len(ground_truth[0]):
    # if is_majority_german_words(answer_text):
        sentences = []
        count = 0
        for item in ground_truth:
            if len(item) >= 2 and item[:2] == "S ":
                count += 1

        if count > 1:
            sentence_str_list = split_str_into_sentences(answer_text)
            for sentence in sentence_str_list:
                sentence.replace("\n"," ")
                words = split_sentences_into_word(sentence)
                sentences.append(' '.join(words))
        else:
            answer_text.replace("\n"," ")
            words = split_sentences_into_word(answer_text)
            sentences.append(" ".join(words))
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Answer: {sentences}")
        p, r, f = cal_m2scorer(sentences, ground_truth)
        print(f"\n[Content Validation]")
        if p == 1:
            print("  Content validation: FULL MATCH")
        else:
            print("  Content validation: MISMATCH")

        answer_score = f
    else:
        answer_score = -1
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    # total_score = 0.1 * format_score + 0.9 * answer_score
    total_score = answer_score
    print("\n" + "-" * 80)
    print(f" Final Score ".center(80, '-'))
    # print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("=" * 80 + "\n")

    return total_score