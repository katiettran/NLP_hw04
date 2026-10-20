
import re
from stop_list import closed_class_stop_words
from nltk.stem import PorterStemmer

STOP_WORDS = set(closed_class_stop_words)
stemmer = PorterStemmer()


def tokenize(text):
    """Lowercase, remove stop words, and apply Porter stemming."""
    words = re.findall(r"[a-z]+", text.lower())

    return [
        stemmer.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]

def load_queries(filename):
    queries = {}
    words = []
    in_query = False
    found_query = False

    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line.startswith(".I "):
                if found_query:
                    queries[len(queries) + 1] = tokenize(" ".join(words))

                words = []
                found_query = True
                in_query = False

            elif line == ".W":
                in_query = True

            elif in_query:
                words.append(line)

    if found_query:
        queries[len(queries) + 1] = tokenize(" ".join(words))

    return queries


def load_documents(filename):
    documents = {}
    current_id = None
    words = []
    collecting = False

    with open(filename, encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line.startswith(".I "):
                if current_id is not None:
                    documents[current_id] = tokenize(" ".join(words))

                current_id = int(line[3:].strip())
                words = []
                collecting = False

            elif line in (".T", ".W"):
                collecting = True

            elif line.startswith("."):
                collecting = False

            elif collecting:
                words.append(line)

    if current_id is not None:
        documents[current_id] = tokenize(" ".join(words))

    return documents


from collections import Counter
import math


def calculate_idf(collection):
    """Calculate IDF using the number of documents containing each term."""
    total = len(collection)
    document_frequency = Counter()

    for tokens in collection.values():
        document_frequency.update(set(tokens))

    return {
        word: math.log((total + 1) / (frequency + 1)) + 1
        for word, frequency in document_frequency.items()
    }


def tfidf_vector(tokens, idf):
    """Create a TF-IDF vector using logarithmic term frequency."""
    counts = Counter(tokens)

    return {
        word: (1 + math.log(count)) * idf[word]
        for word, count in counts.items()
        if word in idf
    }



def cosine_similarity(query_vector, document_vector):
    """Calculate cosine similarity using full TF-IDF vectors."""

    dot_product = sum(
        weight * document_vector.get(word, 0)
        for word, weight in query_vector.items()
    )

    query_norm = math.sqrt(
        sum(weight ** 2 for weight in query_vector.values())
    )

    document_norm = math.sqrt(
        sum(weight ** 2 for weight in document_vector.values())
    )

    if query_norm == 0 or document_norm == 0:
        return 0.0

    return dot_product / (query_norm * document_norm ** 0.75)

def rank_documents(queries, documents):
    """Rank all 1,400 documents for every query."""

    query_idf = calculate_idf(queries)
    document_idf = calculate_idf(documents)

    document_vectors = {
        doc_id: tfidf_vector(tokens, document_idf)
        for doc_id, tokens in documents.items()
    }

    with open("output.txt", "w", encoding="utf-8") as output:
        for query_id, query_tokens in queries.items():
            query_vector = tfidf_vector(query_tokens, query_idf)

            rankings = []

            for doc_id, document_vector in document_vectors.items():
                score = cosine_similarity(query_vector, document_vector)
                rankings.append((doc_id, score))

            rankings.sort(key=lambda item: (-item[1], item[0]))

            for doc_id, score in rankings:
                output.write(f"{query_id} {doc_id} {score:.6f}\n")

    print("Created output.txt successfully!")


def load_relevance(filename):
    """Read Cranfield relevance judgments."""
    relevant = {}

    with open(filename, encoding="utf-8") as file:
        for line in file:
            parts = line.split()

            if len(parts) < 3:
                continue

            query_id, doc_id, grade = map(int, parts[:3])

            if grade > 0:
                relevant.setdefault(query_id, set()).add(doc_id)

    return relevant


def evaluate_map(output_file, relevance_file):
    """Compute standard mean average precision."""
    relevant = load_relevance(relevance_file)
    rankings = {}

    with open(output_file, encoding="utf-8") as file:
        for line in file:
            query_id, doc_id, score = line.split()
            rankings.setdefault(int(query_id), []).append(int(doc_id))

    average_precisions = []

    for query_id, ranked_docs in rankings.items():
        relevant_docs = relevant.get(query_id, set())

        if not relevant_docs:
            continue

        hits = 0
        precision_sum = 0.0

        for rank, doc_id in enumerate(ranked_docs, start=1):
            if doc_id in relevant_docs:
                hits += 1
                precision_sum += hits / rank

        average_precisions.append(
            precision_sum / len(relevant_docs)
        )

    return sum(average_precisions) / len(average_precisions)


if __name__ == "__main__":
    queries = load_queries("cran.qry")
    documents = load_documents("cran.all.1400")

    print("Queries loaded:", len(queries))
    print("Documents loaded:", len(documents))
    print("First query:", queries[1][:15])
    print("First document:", documents[1][:15])
    rank_documents(queries, documents)
    print(f"Baseline MAP: {evaluate_map('output.txt', 'cranqrel'):.4f}")
