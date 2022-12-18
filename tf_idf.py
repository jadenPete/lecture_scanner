#!/usr/bin/env python

import argparse
import bs4
import collections
import json
import math
import os
import re
import sklearn.svm
import subprocess

FPS = 1

DOCUMENT_DELIMETER = "----"

INCLUDED_PROPORTION = 0.3

parser = argparse.ArgumentParser(
	prog="ITSC 2214 Lecture Scanner",
	description="Uses OCR to output perceptual transcripts for a given lecture."
)

parser.add_argument("filename")
parser.add_argument("-l", "--limit",
	help="Only process a given number of frames.",
	type=int
)

parser.add_argument("-r", "--reuse",
	action="store_const",
	const=True,
	default=False,
	help="Attempt to reuse existing frames from the cache."
)

parser.add_argument("-w", "--weights",
	help="A JSON file containing the tf-idfs of included and excluded frame lines."
)

args = parser.parse_args()

def generate_corpus():
	cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cache")

	try:
		os.mkdir(cache_dir)
	except FileExistsError:
		pass

	if not args.reuse:
		for filename in os.listdir(cache_dir):
			os.remove(os.path.join(cache_dir, filename))

	if not args.reuse or len(os.listdir(cache_dir)) == 0:
		# https://stackoverflow.com/a/28321986
		subprocess.call([
			"ffmpeg",
			"-i",
			args.filename,
			"-filter:v",
			f"fps={FPS}",
			os.path.join(cache_dir, "%d.png")
		])

	corpus = []

	i = 1

	while args.limit is None or i <= args.limit:
		if not os.path.exists(frame_path := os.path.join(cache_dir, f"{i}.png")):
			break

		if not os.path.exists(document_path := os.path.join(cache_dir, f"{i}.json")):
			output = subprocess.check_output(["tesseract", frame_path, "stdout", "hocr"]).decode()

			soup = bs4.BeautifulSoup(output, features="html.parser")

			result = []

			for line in soup.find_all(class_="ocr_line"):
				word_tags = line.find_all(class_="ocrx_word")

				result.append({
					"confidence": sum(
						int(re.search(r"(?<=x_wconf )\d+", word["title"])[0]) / 100

						for word in word_tags
					) / len(word_tags),

					"words": [word.string for word in word_tags]
				})

			with open(document_path, "w") as file:
				json.dump(result, file)

		with open(document_path) as file:
			corpus.append(json.load(file))

		i += 1

	return corpus

def print_corpus_tf_idfs(corpus):
	for document, document_tf_idf in zip(corpus, tf_idf_by_line(corpus)):
		output = []

		for line, tf_idf in zip(document, document_tf_idf):
			output.append([str(tf_idf), " ".join(line["words"])])

		if len(output) > 0:
			tf_idf_width = max(len(line[0]) for line in output)

		for line in output:
			line[0] = line[0] + " " * (tf_idf_width - len(line[0]))

			print(*line)

		print(DOCUMENT_DELIMETER)

def print_corpus_transcript(corpus, model):
	result = []

	for i, (document, document_tf_idf) in enumerate(zip(corpus, tf_idf_by_line(corpus))):
		lines = []

		result.append({
			"start": i / FPS,
			"end": (i + 1) / FPS,
			"lines": lines
		})

		for line, tf_idf in zip(document, document_tf_idf):
			if model.predict([[tf_idf, line["confidence"]]])[0]:
				lines.append(" ".join(line["words"]))

	print(json.dumps(result, indent="\t"))

def tf_idf_by_line(corpus):
	idf = collections.defaultdict(int)

	for document in corpus:
		for word in {word for line in document for word in line["words"]}:
			idf[word] += 1

	for word in idf:
		idf[word] = math.log(len(corpus) / idf[word])

	for document in corpus:
		tf = collections.defaultdict(int)

		for word in (words := [word for line in document for word in line["words"]]):
			tf[word] += 1

		for word in tf:
			tf[word] /= len(words)

		result = []

		for line in document:
			word_count = len(line["words"])

			if word_count == 0:
				line_tf_idf = 0
			else:
				line_tf_idf = sum(tf[word] * idf[word] for word in line["words"]) / word_count

			result.append(line_tf_idf * line["confidence"])

		yield result

def tf_idf_model(corpus):
	if args.weights is None:
		return

	with open(args.weights) as file:
		weights = json.load(file)

	model = sklearn.svm.SVC(class_weight={
		False: 1 / (2 * (1 - INCLUDED_PROPORTION)),
		True: 1 / (2 * INCLUDED_PROPORTION)
	})

	x, y = [], []

	for document, document_tf_idf, document_weights in zip(corpus, tf_idf_by_line(corpus), weights):
		for line, tf_idf, is_included in zip(document, document_tf_idf, document_weights):
			x.append([tf_idf, line["confidence"]])
			y.append(is_included)

	model.fit(x, y)

	return model

corpus = generate_corpus()

model = tf_idf_model(corpus)

if model is None:
	print_corpus_tf_idfs(corpus)
else:
	print_corpus_transcript(corpus, model)
