import re
from nltk.tokenize import word_tokenize
import docx2txt
import PyPDF2
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from django.http import JsonResponse

# Load the pre-trained model
model = Word2Vec.load("word2vec_model.model")


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return tokens


def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text


def read_file(file):
    if file.name.endswith('.docx'):
        return docx2txt.process(file)
    elif file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    else:
        raise ValueError('Unsupported file format. Please use .docx or .pdf files.')


def average_word_vectors(words, model, num_features):
    feature_vector = np.zeros((num_features,), dtype="float32")
    nwords = 0
    vocabulary = set(model.wv.key_to_index)

    for word in words:
        if word in vocabulary:
            nwords += 1
            feature_vector = np.add(feature_vector, model.wv[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def calculate_similarity(job_text, resume_text):
    job_requirement = preprocess_text(job_text)
    resume_tokens = preprocess_text(resume_text)

    num_features = model.vector_size
    job_vector = average_word_vectors(job_requirement, model, num_features)
    resume_vector = average_word_vectors(resume_tokens, model, num_features)

    similarity_score = cosine_similarity([job_vector], [resume_vector])[0][0]

    return similarity_score * 100


class SimilarityView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    def post(self, request, format=None):
        print("posttttttttt")
        job_text = request.data.get('job_text')
        resume_file = request.FILES.get('resume_file')

        if not job_text or not resume_file:
            return Response({"error": "Job text and resume file are required"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            resume_text = read_file(resume_file)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        similarity_score = calculate_similarity(job_text, resume_text)
        return JsonResponse({"similarity_score": similarity_score})
