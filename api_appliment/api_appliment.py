import os
import random
import json
import time
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import RequestException
from threading import Lock
from pydantic import BaseModel
import pandas
import re
from openai import OpenAI
from dotenv import load_dotenv
from utils import split_sentence, correct_and_conjunction_sentence, article_process, split_sentence_align

# Load environment variables from the .env file
load_dotenv()

client = OpenAI(
    api_key="",
    base_url=""
)

instruction = [
    """
You are an expert in German language education. I will provide you with a student-written German essay along with a rough explanation of its errors. Please act as a teacher and give the student more detailed, constructive feedback based on the provided information.

Your feedback must include:
1. The erroneous sentences from the article.
2. Identification of the error type and guidance to help the student reflect (determine the precise error type based on the sentence itself—not just repeat the rough explanation—and then guide the student).
3. How to correctly revise the erroneous sentence, along with an explanation of why the revision is correct.

You must separate these three parts clearly and label them as follows:
<error> All erroneous sentences from the article </error>;
<think> Identify the error type and guide the student to reflect </think>;
<correct> How to revise the erroneous sentence correctly and why </correct>;

These three sections must not overlap or encroach on each other. Respond in English. Use encouraging and empathetic language—for example, start with a brief positive evaluation of the essay and words of encouragement, and conclude with a summary that motivates the student. Also, modify the phrasing of the opening and closing remarks in the example.

Example:
"Ich bin ein aktiver Schüler und Ich mache gern Sport. Mein Lieblingssprot ist Fußball. Ich habe ein Fußballspiel von Deutschland mit Agentina gesehen. Danach mag ich Fußball sehr. Ich sehe oft Fußballspiele und spiele manchmal Fußball. Ich mag BVB und Marco Reus, sie sind meine echte Liebe. Und ich möchte ewig Fußball mögen."

You have clearly expressed your passion for football and illustrated your interest with concrete examples. Writing is a journey of continuous improvement, and you've already taken an important step! Next, I’ll point out specific areas for refinement to help you further enhance your German writing skills.

<error>
"Ich bin ein aktiver Schüler und Ich mache gern Sport." is incorrect.
"Mein Lieblingssprot ist Fußball." is incorrect.
"Ich habe ein Fußballspiel von Deutschland mit Agentina gesehen." is incorrect.
"Ich mag BVB und Marco Reus, sie sind meine echte Liebe." is incorrect.
</error>

<think>
"Ich bin ein aktiver Schüler und Ich mache gern Sport.": In this sentence, “Ich” appears twice, and the second instance is incorrectly capitalized. In German grammar, only the first word of a sentence should be capitalized unless it’s a proper noun. Can you think about how capitalization works within compound sentences?
"Mein Lieblingssprot ist Fußball.": The word “Lieblingssprot” is misspelled. Do you recall how to correctly spell “favorite sport” in German?
"Ich habe ein Fußballspiel von Deutschland mit Agentina gesehen.": This sentence uses the wrong prepositions (“von” and “mit”), and “Agentina” is misspelled. When describing a match between two countries, which preposition should be used? Also, what’s the correct German spelling for “Argentina”?
"Ich mag BVB und Marco Reus, sie sind meine echte Liebe.": A definite article is missing before “BVB,” and punctuation is used incorrectly. When referring to a specific football club like BVB, do we need a definite article in German? Also, when connecting two independent but related clauses, which punctuation mark is most appropriate?
</think>

<correct>
"Ich bin ein aktiver Schüler und Ich mache gern Sport." should be revised to "Ich bin ein aktiver Schüler und ich mache gern Sport." In German, “ich” is capitalized only at the beginning of a sentence. Within a sentence, it should be lowercase.
"Mein Lieblingssprot ist Fußball." should be corrected to "Mein Lieblingssport ist Fußball." The word “Sport” was misspelled as “Sprot.” The correct spelling is “Sport.”
"Ich habe ein Fußballspiel von Deutschland mit Agentina gesehen." should be changed to "Ich habe ein Fußballspiel Deutschland gegen Argentinien gesehen." The preposition “von...mit” is inappropriate here; “gegen” is used to indicate teams competing against each other. Additionally, “Argentinien” is the correct German spelling for “Argentina.”
"Ich mag BVB und Marco Reus, sie sind meine echte Liebe." can be improved to "Ich mag den BVB und Marco Reus. Sie sind meine echte Liebe." In German, specific football clubs like BVB usually require a definite article (“den”). Also, splitting the original sentence into two improves clarity and grammatical correctness.
</correct>

In summary, your essay is well-structured and full of enthusiasm—this is truly commendable! However, attention to detail—such as spelling, punctuation, and grammar—will help you reach the next level. I recommend proofreading your work carefully after writing, especially for proper nouns and punctuation. Keep up the great work—you’re doing wonderfully!
"""
]

train_folder = "../tag_excel_process_2899/train_folder"
json_path = "../explanation_for_pretrain/explanation_for_pretrain_2899_EN.json"

explanation_to_tag = {
    "WS": "Spelling error",
    "GKS": "Incorrect capitalization (uppercase/lowercase)",
    "GZS": "Words that should be written together are separated, or vice versa",
    "ST": "Incorrect syllable division",
    "ZS": "Punctuation error: wrong punctuation, missing punctuation, or misplaced punctuation",
    "Flex": "Morphological error: incorrect inflection of nouns (number/case), adjectives (gender/number/case/strong-weak declension), verbs (conjugation), or articles",
    "Aux": "Wrong auxiliary verb used for tense or voice",
    "Kompb": "Incorrect comparative or superlative form",
    "Wortb": "Over-application of word formation rules, resulting in non-existent German words",
    "StV": "Incorrect verb position in main clause",
    "StVNS": "Incorrect verb position in subordinate clause",
    "StPTKVZ": "Incorrect placement or omission of separable verb prefix",
    "StW": "Incorrect question word position",
    "StPTK": "Incorrect placement of negation or focus particles",
    "StMF": "Incorrect word order in the midfield (Mittelfeld)",
    "StN": "Incorrect word order within noun phrases",
    "StND": "Incorrect order in names, dates, or place expressions",
    "HSt": "Incorrect word order in emphasized constructions",
    "ValV": "Verb valency error: wrong case after verb, missing/wrong preposition, or incorrect prepositional complement",
    "ValVRefl": "Reflexive pronoun missing, redundant, or incorrect (including case/number errors)",
    "ValN": "Incorrect preposition governed by a noun",
    "ValADJ": "Incorrect preposition governed by an adjective",
    "ValAP": "Prepositional valency error: wrong case or number after preposition",
    "ValWP": "Directional preposition (e.g., in, auf, unter) used with wrong case when indicating location",
    "SuppLok": "Incorrect or missing preposition in locative expressions",
    "SuppTemp": "Incorrect, redundant, or missing preposition in temporal expressions",
    "SuppMod": "Modal adverbial expression used incorrectly",
    "SuppKaus": "Causal adverbial expression used incorrectly",
    "KorKomp": "Correlative for complement clause missing, wrong, or redundant",
    "KorSupp": "Correlative for supplement clause missing, wrong, or redundant",
    "KonKOU": "Subordinating conjunction missing, wrong, or redundant",
    "KonKOUI": "zu-construction subordinator missing, wrong, or redundant",
    "KonKON": "Coordinating conjunction missing, wrong, or redundant",
    "KonREL": "Relative pronoun or adverb missing or incorrect",
    "KongrSubj": "Lack of agreement between subject and predicate in gender/number/case",
    "KongrAnt": "Lack of agreement with antecedent in gender/number/case",
    "Subj": "Subject missing or redundant",
    "Präd": "Predicate missing or redundant",
    "Expl": "Expletive 'es' (formal subject) missing or redundant",
    "PTKVZ": "Separable verb prefix missing or incorrect",
    "Attr": "Extended attributive construction error; wrong use of present/past participles as attributes",
    "Infin": "Incorrect use or formation of infinitives",
    "Quant": "Quantifier missing or incorrect",
    "Det": "Determiner redundant or missing",
    "Poss": "Possessive form missing or incorrect",
    "Komp": "Comparative particle (als, wie) missing, wrong, or redundant; incorrect comparison object or superlative form",
    "Neg": "Incorrect negation form",
    "Gen": "Incorrect gender assignment",
    "GenFW": "Incorrect gender assignment for loanwords",
    "LexV": "Wrong verb choice",
    "LexMV": "Wrong modal verb choice",
    "LexN": "Wrong noun choice",
    "LexAP": "Wrong preposition choice",
    "LexADJ": "Wrong adjective choice",
    "LexADV": "Wrong adverb choice",
    "LexPRON": "Wrong pronoun choice",
    "LexZ": "Wrong numeral choice",
    "Phr": "Fixed phrase used incorrectly",
    "Def": "Inappropriate definiteness based on context",
    "Num": "Semantically inappropriate number (singular/plural)",
    "Sexus": "Gender mismatch (male/female reference error)",
    "Ant": "Antecedent missing or incorrect",
    "Mod": "Wrong mood (indicative, subjunctive, imperative)",
    "Temp": "Inappropriate tense based on context",
    "GenV": "Wrong voice (active/passive)"
}

type = {
    "FehlerOrth": """WS(Wortschreibung): Spelling errors e.g. missing letters, redundancy, misspellings;       
                     GKS(Groß- und Kleinschreibung): Incorrect case choices for initial letters, including common nouns, adjectival variations of nouns;       
                     GZS(Getrennt- und Zusammenschreibung): Disconnecting words where they shouldn't be disconnected and not disconnecting them where they should be disconnected;       
                     ST(Silben­trennung): Faulty syllable divisions, or lack of syllable divisions;       
                     ZS(Zeichen­setzung): Wrong choice of punctuation, misplaced punctuation, missing or redundant punctuation;
                  """,
    "FehlerMorph": """Flex(Flexionsklasse): Errors in changing noun forms, verb simple forms, adjective simple forms, and article-final forms;       
                      Aux(Auxiliar): Incorrect choice of auxiliary verbs (haben, werden, sein) when expressing tense and voice;       
                      Kompb(Komparativbildung): Comparative or supreme forms change incorrectly;       
                      Wortb(Wortbildung): Errors in word formation, e.g. in the case of inappropriate root and affix combinations or overuse of grammatical rules to produce a wrong word that does not exist in German;
                   """,
    "FehlerSyn": """StV(Stellung des Verbs im Hauptsatz): Incorrect placement of the verb in the main clause;       
                    StVNS(Stellung des Verbs im Nebensatz): Special reference to incorrect verb position in subordinate clauses;       
                    StPTKVZ(Stellung der Verbpartikel): Separable verbs with incorrectly placed separable prefixes, including unseparated or incorrectly segregated prefixes;       
                    StW(Stellung des Frageworts): Incorrect placement of question words;       
                    StPTK(Stellung der Negations- und Fokuspartikel): The negatives are misplaced;       
                    StMF(Stellung im Mittelfeld): Components in the centre of the field are misplaced;       
                    StN(Stellung in der Nominalgruppe): The placement of the elements in the noun phrase is wrong;       
                    StND(Stellung in Namens- oder Datumsangaben): Incorrect placement of name, date, place;       
                    HSt(Stellung in Herausstellung): Incorrect order within the emphasis or incorrect order of the reference sentence due to shifting of the emphasis;       
                    ValV(Verbvalenz): Incorrect verb conjugation;       
                    ValVRefl(Reflexivpronomen in der Verbvalenz): Failure to use or incorrect use of reflexive pronouns when they should be used in verb conjugation;       
                    ValN(Substantivvalenz): Noun valency collocation is incorrect;       
                    ValADJ(Adjektivvalenz): The adjective valency collocation is incorrect;       
                    ValAP(Adpositionsvalenz): error in prepositional valency collocation;       
                    ValWP(Wechselpräpositionsvalenz): Specific directional prepositions (including in, über, auf, unter, an, neben, zwischen, vor, hinter) when paired with a noun to indicate place location;       
                    SuppLok(Supplement (lokal)): Incorrect use of place descriptors, such as wrong choice of words for place descriptors or lack of prepositions in the parts of the place descriptors;       
                    SuppTemp(Supplement (temporal)): Incorrect use of time descriptors, e.g., time descriptors are redundant or incorrectly chosen, prepositions are missing from the part of the table where the time descriptor is given;       
                    SuppMod(Supplement (modal)): Incorrect use of modal descriptors, e.g. wrong choice of words;       
                    SuppKaus(Supplement (kausal)): Incorrect use of cause specifiers;       
                    KorKomp(Korrelat zu einem Komplementsatz): Missing, incorrect, or redundant correlative of the referring complement clause;       
                    KorSupp(Korrelat zu einem Supplementsatz): Missing, incorrect, or redundant linking words referring to explanatory sentences;       
                    KonKOU(Subjunktion): missing or incorrect subject-subject connective;       
                    KonKON(Konjunktion): Missing or incorrect connectors of parallel elements;       
                    KonREL(Retativwort): Missing or incorrect relational pronouns or relational adverbs;       
                    KongrSubj(Kongruenz mit dem Subjekt): The predicate does not agree with the subject in the personal gendered number case;       
                    KongrAnt(Kongruenz mit dem Antezedens): Failure to agree with the antecedent in the gendered number case;       
                    Subj(Subjekt): Missing, redundant, incorrect subject;       
                    Prä(Prädikat): Missing or redundant formal subjects;       
                    Infin((In)Finitheit): Incorrect use of infinitives, incorrect formation of infinitives and failure to use the prototype for verbs after modal verbs;       
                    Expl(Expletivum/Vorfeld-'es'): Missing or redundant formal subjects;       
                    Quant(Quantifikation): Missing, incorrect or redundant quantifiers;       
                    Det(Determinator): superfluous and missing qualifiers;       
                    Poss(Possessivität): possessive error;       
                    Komp(Komparations-, Vergleichspartikel): Missing, incorrect or redundant comparatives (Vergleichspartikel: als, wie);       
                    Neg(Negation): negative form error; """,
    "FehlerLex": """Gen(Genus): error of judgment in word formation;       
                    GenFW(Genus von Fremdwörtern): Incorrect lexical judgement of foreign nouns;       
                    Grundf(Grundform): ;       
                    Lex(Wortwahl): incorrect choice of words;       
                    Phr(Wendung): fixed expression error;""",
    "FehlerSem": """Def(Definitheit): Semantically inappropriate choice of definiteness based on context;       
                    Num(Numerus): Semantically inappropriate singular and plural;       
                    Sexus(Sexus/konzep­tuelles Genus): Misgendering men and women in the context of a sentence;       
                    Ant(Antezedens): Unnecessary, faulty, missing antecedents/clauses;       
                    Mod(Modus): Incorrect choice of verbs in the direct, virtual and imperative forms;       
                    Temp(Tempus): Semantically inappropriate tenses;       
                    GenV(Genus verbi): Misapplication of active and passive dynamic choices;
                 """,
    "FehlerSyn(Wortstellung)": """StV(Stellung des Verbs im Hauptsatz): Incorrect placement of the verb in the main clause;       
                                   StVNS(Stellung des Verbs im Nebensatz): Special reference to incorrect verb position in subordinate clauses;       
                                   StPTKVZ(Stellung der Verbpartikel): Separable verbs with incorrectly placed separable prefixes, including unseparated or incorrectly segregated prefixes;       
                                   StW(Stellung des Frageworts): Incorrect placement of question words;       
                                   StPTK(Stellung der Negations- und Fokuspartikel): The negatives are misplaced;       
                                   StMF(Stellung im Mittelfeld): Components in the centre of the field are misplaced;       
                                   StN(Stellung in der Nominalgruppe): The placement of the elements in the noun phrase is wrong;       
                                   StND(Stellung in Namens- oder Datumsangaben): Incorrect placement of name, date, place;       
                                   HSt(Stellung in Herausstellung): Incorrect order within the emphasis or incorrect order of the reference sentence due to shifting of the emphasis;
                                """,
    "FehlerSyn(Valenz)": """ValV(Verbvalenz): Incorrect verb conjugation;       
                            ValVRefl(Reflexivpronomen in der Verbvalenz): Failure to use or incorrect use of reflexive pronouns when they should be used in verb conjugation;       
                            ValN(Substantivvalenz): Noun valency collocation is incorrect;       
                            ValADJ(Adjektivvalenz): The adjective valency collocation is incorrect;       
                            ValAP(Adpositionsvalenz): error in prepositional valency collocation;       
                            ValWP(Wechselpräpositionsvalenz): Specific directional prepositions (including in, über, auf, unter, an, neben, zwischen, vor, hinter) when paired with a noun to indicate place location;
                         """,
    "FehlerSyn(Supplement)": """SuppLok(Supplement (lokal)): Incorrect use of place descriptors, such as wrong choice of words for place descriptors or lack of prepositions in the parts of the place descriptors;       
                                 SuppTemp(Supplement (temporal)): Incorrect use of time descriptors, e.g., time descriptors are redundant or incorrectly chosen, prepositions are missing from the part of the table where the time descriptor is given;       
                                 SuppMod(Supplement (modal)): Incorrect use of modal descriptors, e.g. wrong choice of words;       
                                 SuppKaus(Supplement (kausal)): Incorrect use of cause specifiers;
                              """,
    "FehlerSyn(Korrelat)": """KorKomp(Korrelat zu einem Komplementsatz): Missing, incorrect, or redundant correlative of the referring complement clause;       
                               KorSupp(Korrelat zu einem Supplementsatz): Missing, incorrect, or redundant linking words referring to explanatory sentences;       
                               KonKOU(Subjunktion): missing or incorrect subject-subject connective;       
                               KonKON(Konjunktion): Missing or incorrect connectors of parallel elements;       
                               KonREL(Retativwort): Missing or incorrect relational pronouns or relational adverbs;       
                               KongrSubj(Kongruenz mit dem Subjekt): The predicate does not agree with the subject in the personal gendered number case;       
                               KongrAnt(Kongruenz mit dem Antezedens): Failure to agree with the antecedent in the gendered number case;
                            """,
    "FehlerSyn(Andere)": """Subj(Subjekt): Missing, redundant, incorrect subject;       
                            Prä(Prädikat): Missing or redundant formal subjects;       
                            Infin((In)Finitheit): Incorrect use of infinitives, incorrect formation of infinitives and failure to use the prototype for verbs after modal verbs;       
                            Expl(Expletivum/Vorfeld-'es'): Missing or redundant formal subjects;       
                            Quant(Quantifikation): Missing, incorrect or redundant quantifiers;       
                            Det(Determinator): superfluous and missing qualifiers;       
                            Poss(Possessivität): possessive error;       
                            Komp(Komparations-, Vergleichspartikel): Missing, incorrect or redundant comparatives (Vergleichspartikel: als, wie);       
                            Neg(Negation): negative form error;
                         """
}

# (The rest of the functions remain unchanged as they contain no Chinese text.)
def reassemble_text(tokens):
    result = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # If encountering a quotation mark, collect quoted content as a single unit
        if token == '"':
            quoted_sentence = []
            i += 1
            while i < len(tokens) and tokens[i] != '"':
                quoted_sentence.append(tokens[i])
                i += 1
            if quoted_sentence:
                result.append(f'"{join_tokens(quoted_sentence)}"')
            i += 1  # Skip closing quote
        else:
            result.append(token)
            i += 1
    return join_tokens(result)

def join_tokens(tokens):
    text = ''
    for token in tokens:
        if token in ',.!?;:)]}' and len(text) > 0 and text[-1] != ' ':
            text += token
        elif token in '([{' or (len(text) > 0 and text[-1] == ' '):
            text += token
        else:
            text += ' ' + token
    return text.lstrip()

def conjunction_sentence(sentence):
    candidate = ""
    for index in range(len(sentence["origin"])):
        if sentence["origin"][index] != "" and sentence["origin"][index] not in string.punctuation and index != 0:
            candidate = candidate + " " + sentence["origin"][index]
        elif sentence["origin"][index] not in string.punctuation:
            candidate += sentence["origin"][index]
        else:
            candidate = candidate + sentence["origin"][index]
    return candidate

def conjunction_correct_sentence(sentence):
    candidate = ""
    for index in range(len(sentence["correct"])):
        if sentence["correct"][index] != "" and sentence["correct"][index] not in string.punctuation and index != 0:
            candidate = candidate + " " + sentence["correct"][index]
        elif sentence["correct"][index] not in string.punctuation:
            candidate += sentence["correct"][index]
        else:
            candidate = candidate + sentence["correct"][index] + " "
    return candidate

def construct_json(file_name, sentence, response):
    origin_sentence = conjunction_sentence(sentence)
    message = {
        "file_name": file_name,
        "article": origin_sentence,
        "description": response
    }
    return message

def get_completion_from_messages(messages, retries=3):
    backoff_factor = 1
    for attempt in range(1, retries + 2):
        try:
            chat_completion = client.chat.completions.create(
                temperature=0,
                messages=messages,
                model="gpt-4o",
            )
            return chat_completion.choices[0].message.content
        except RequestException as e:
            print(f"Attempt {attempt} failed with error: {e}")
            if attempt > retries:
                raise
            wait_time = backoff_factor * (2 ** (attempt - 1))
            jitter = random.uniform(0.5, 1.5)
            sleep_duration = wait_time * jitter
            print(f"Retrying in {sleep_duration:.2f} seconds...")
            time.sleep(sleep_duration)

def multi_sentence_explanation(file_name):
    json_message = []
    prompt = instruction[0]
    system_message = {
        "role": "system",
        "content": prompt
    }
    type_pattern = r'(?<=:).*?(?= |$)'
    sentence_list = split_sentence(os.path.join(train_folder, file_name))
    for i in range(len(sentence_list)):
        if i == 0:
            sentence_group = sentence_list[i:i + 2]
        else:
            sentence_group = sentence_list[i - 1:i + 2]
        sentence = {
            "origin": [],
            "type": [],
            "correct": []
        }
        for cur_sentence in sentence_group:
            sentence["origin"] += cur_sentence["origin"]
            sentence["type"] += cur_sentence["type"]
            sentence["correct"] += cur_sentence["correct"]
        origin_sentence = conjunction_sentence(sentence)
        input = f"Here is a German text: '{origin_sentence}'\nThis text contains some lexical or grammatical errors. Below is a rough description of the errors:"
        description = ""
        for index in range(len(sentence["type"])):
            type_str = sentence["type"][index]
            if type_str != "":
                word = sentence["origin"][index]
                correct = sentence["correct"][index]
                st_flag = False
                # Error description
                if word != "" and word not in string.punctuation:
                    description += f"The word '{word}' in the sentence is incorrect because "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for type_str in type_str_list:
                        type_list.extend(type_str.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            if error.startswith("St"):
                                st_flag = True
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for error in type_list:
                            if not error.endswith("-"):
                                if error.startswith("St"):
                                    st_flag = True
                                description += f"'{explanation_to_tag[error]}'."
                                if error != type_list[-1]:
                                    description += " and "
                                else:
                                    description += "."
                elif word != "" and word in string.punctuation:
                    description += f"The punctuation mark '{word}' in the sentence is incorrect because "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for type_str in type_str_list:
                        type_list.extend(type_str.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            if error.startswith("St"):
                                st_flag = True
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for error in type_list:
                            if not error.endswith("-"):
                                if error.startswith("St"):
                                    st_flag = True
                                description += f"'{explanation_to_tag[error]}'."
                                if error != type_list[-1]:
                                    description += " and "
                                else:
                                    description += "."
                else:
                    # Missing word case
                    next_word = sentence["origin"][index] if index < len(sentence["origin"]) else ""
                    description += f"A word is missing before '{next_word}', and the error is due to "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for type_str in type_str_list:
                        type_list.extend(type_str.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for error in type_list:
                            if not error.endswith("-"):
                                description += f"'{explanation_to_tag[error]}'"
                                if error != type_list[-1]:
                                    description += " and "
                                else:
                                    description += "."
                # Append correction suggestion
                if correct != "" and not st_flag:
                    description += f" It should be corrected to '{correct}'."
                elif not st_flag:
                    description += " It should be deleted."
                else:
                    description += " It should be moved to the correct position."
                input += f" {description};"
        # Only proceed if there are actual errors described
        if description.strip():
            correct_sentence = conjunction_correct_sentence(sentence)
            input += f" Corrected sentence: '{correct_sentence}'"
            user_message = {
                "role": "user",
                "content": input
            }
            message = [system_message, user_message]
            response = get_completion_from_messages(message)
            json_message.append(construct_json(file_name, sentence, response))
    return json_message

def conjunction_article(sentence_list):
    article = ""
    for sentence in sentence_list:
        article += conjunction_sentence(sentence)
    return article

def sentence_correct(sentence):
    instruction = """You are a German writing expert. Below, I will give you an incorrect sentence and a reference-corrected sentence.
Your task is to act as a student revising the sentence: first, identify the error and think about how to fix it inside <think></think>, then reply with ONLY the corrected sentence inside <answer></answer>."""
    origin = reassemble_text(sentence["origin"])
    correct = reassemble_text(sentence["correct"])
    return [{
        "role": "system",
        "content": instruction,
    },
        {
            "role": "user",
            "content": f"Incorrect sentence: \"{origin}\"\nCorrect sentence: \"{correct}\"",
        }]

def article_correct(file_name):
    instruction = """You are a German writing expert. Below, I will give you an incorrect essay and a reference-corrected version.
Your task is to act as a student revising the text: first, identify the errors and think about how to fix them inside <think></think>, then reply with ONLY the corrected essay inside <answer></answer>."""
    article_list = article_process(os.path.join(train_folder, file_name))
    origin = reassemble_text(article_list["origin"])
    correct = reassemble_text(article_list["correct"])
    return [{
        "role": "system",
        "content": instruction,
    },
        {
            "role": "user",
            "content": f"Incorrect essay: \"{origin}\"\nCorrect essay: \"{correct}\"",
        }]


def article_explanation(file_name):
    json_message = []
    prompt = instruction[0]
    system_message = {
        "role": "system",
        "content": prompt
    }

    type_pattern = r'(?<=:).*?(?= |$)'
    article_list = article_process(os.path.join(train_folder, file_name))
    sentence_list = split_sentence(os.path.join(train_folder, file_name))
    origin_article = conjunction_sentence(article_list)
    input_text = f"Text:\n\"{origin_article}\"\nRough explanation:\n"

    for sentence in sentence_list:
        origin_sentence = conjunction_sentence(sentence)
        sentence_input = f"In the sentence: '{origin_sentence}': "
        description = ""
        index = 0
        while index < len(sentence["type"]):
            type_str = sentence["type"][index]
            if type_str != "":
                word = sentence["origin"][index]
                correct = sentence["correct"][index]
                # Merge consecutive tokens with same error type
                while index < len(sentence["type"]) - 1 and type_str == sentence["type"][index + 1]:
                    word += " " + sentence['origin'][index + 1]
                    correct += " " + sentence['correct'][index + 1]
                    index += 1

                st_flag = False
                stt_flag = False

                if word != "" and word not in string.punctuation:
                    description += f"The word '{word}' is incorrect because "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for t in type_str_list:
                        type_list.extend(t.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            if error.startswith("St"):
                                st_flag = True
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for i, error in enumerate(type_list):
                            if not error.endswith("-"):
                                if error.startswith("St"):
                                    st_flag = True
                                description += f"'{explanation_to_tag[error]}'"
                                if i < len(type_list) - 1:
                                    description += " and "
                                else:
                                    description += "."
                elif word != "" and word in string.punctuation:
                    description += f"The punctuation mark '{word}' is incorrect because "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for t in type_str_list:
                        type_list.extend(t.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            if error.startswith("St"):
                                st_flag = True
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for i, error in enumerate(type_list):
                            if not error.endswith("-"):
                                if error.startswith("St"):
                                    st_flag = True
                                description += f"'{explanation_to_tag[error]}'"
                                if i < len(type_list) - 1:
                                    description += " and "
                                else:
                                    description += "."
                else:
                    # Missing word case
                    if index < len(sentence["origin"]) - 1:
                        next_word = sentence["origin"][index + 1]
                    else:
                        next_word = "end of sentence"
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for t in type_str_list:
                        type_list.extend(t.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            description += f"A word is missing before '{next_word}' because "
                            description += f"'{explanation_to_tag[error]}'."
                        else:
                            stt_flag = True
                    else:
                        description += f"A word is missing before '{next_word}' because "
                        for i, error in enumerate(type_list):
                            if not error.endswith("-"):
                                description += f"'{explanation_to_tag[error]}'"
                                if i < len(type_list) - 1:
                                    description += " and "
                                else:
                                    description += "."

                # Suggestion for correction
                if correct != "" and not st_flag and not stt_flag:
                    description += f" It should be corrected to '{correct}'."
                elif not st_flag and not stt_flag:
                    description += " It should be deleted."
                elif not stt_flag:
                    description += " It should be moved to the correct position."

            index += 1

        if description:
            sentence_input += description + "; "
            correct_sentence = conjunction_correct_sentence(sentence)
            sentence_input += f"Corrected sentence: '{correct_sentence}'.\n"
            input_text += sentence_input
        else:
            input_text += f"In the sentence: '{origin_sentence}': This sentence contains no errors.\n"

    user_message = {
        "role": "user",
        "content": input_text
    }

    message = [system_message, user_message]
    return message

def sentence_explanation(file_name):
    prompt = instruction[0]
    system_message = {
        "role": "system",
        "content": prompt
    }
    type_pattern = r'(?<=:).*?(?= |$)'
    sentence_list = split_sentence(os.path.join(train_folder, file_name))
    
    for sentence in sentence_list:
        origin_sentence = conjunction_sentence(sentence)
        input_text = f"Text: '{origin_sentence}'\nRough explanation:\n"
        description = ""
        index = 0
        while index < len(sentence["type"]):
            type_str = sentence["type"][index]
            if type_str != "":
                word = sentence["origin"][index]
                correct = sentence["correct"][index]
                st_flag = False

                if word != "" and word not in string.punctuation:
                    description += f"The word '{word}' is incorrect because "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for t in type_str_list:
                        type_list.extend(t.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            if error.startswith("St"):
                                st_flag = True
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for i, error in enumerate(type_list):
                            if not error.endswith("-"):
                                if error.startswith("St"):
                                    st_flag = True
                                description += f"'{explanation_to_tag[error]}'"
                                if i < len(type_list) - 1:
                                    description += " and "
                                else:
                                    description += "."
                elif word != "" and word in string.punctuation:
                    description += f"The punctuation mark '{word}' is incorrect because "
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for t in type_str_list:
                        type_list.extend(t.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            if error.startswith("St"):
                                st_flag = True
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        for i, error in enumerate(type_list):
                            if not error.endswith("-"):
                                if error.startswith("St"):
                                    st_flag = True
                                description += f"'{explanation_to_tag[error]}'"
                                if i < len(type_list) - 1:
                                    description += " and "
                                else:
                                    description += "."
                else:
                    # Missing word
                    next_word = sentence["origin"][index] if index < len(sentence["origin"]) else "end of sentence"
                    type_list = []
                    type_str_list = re.findall(type_pattern, type_str)
                    for t in type_str_list:
                        type_list.extend(t.split(";"))
                    if len(type_list) == 1:
                        error = type_list[0]
                        if not error.endswith("-"):
                            description += f"A word is missing before '{next_word}' because "
                            description += f"'{explanation_to_tag[error]}'."
                    else:
                        description += f"A word is missing before '{next_word}' because "
                        for i, error in enumerate(type_list):
                            if not error.endswith("-"):
                                description += f"'{explanation_to_tag[error]}'"
                                if i < len(type_list) - 1:
                                    description += " and "
                                else:
                                    description += "."

                # Correction suggestion
                if correct != "" and not st_flag:
                    description += f" It should be corrected to '{correct}'."
                elif not st_flag:
                    description += " It should be deleted."
                else:
                    description += " It should be moved to the correct position."

                input_text += description + "; "
                correct_sentence = conjunction_correct_sentence(sentence)
                input_text += f"Corrected version: '{correct_sentence}'."

            index += 1

        if description:
            user_message = {
                "role": "user",
                "content": input_text
            }
            message = [system_message, user_message]
            return message  # Note: returns first sentence only (original behavior)

    # Fallback if no errors found
    user_message = {
        "role": "user",
        "content": f"Text: '{conjunction_sentence(sentence_list[0])}'\nRough explanation:\nThis sentence contains no errors."
    }
    return [system_message, user_message]

# 线程安全地更新json_message
def process_file(file_name, lock, json_message):
    try:
        json_message.extend(sentence_explanation(file_name))
        json_message.extend(multi_sentence_explanation(file_name))
        json_message.extend(article_explanation(file_name))

        with lock:
            print(file_name)
            with open(json_path, "w", encoding="utf-8") as file:
                file.write(json.dumps(json_message, ensure_ascii=False, indent=1) + "\n")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
