# lecture_scanner
A program for generating optical transcripts for ITSC 2214, Data Structures and Algorithms lectures.

## Scripts

| Filename              | Purpose                                                                            |
|-----------------------|------------------------------------------------------------------------------------|
| display_content_bb.py | A failed experiment to extract the most relevant bounding box from the feed.       |
| tf_idf.py             | Performs OCR on the entire frame, using postprocessing to exclude irrelevant text. |

## How It Works

1. Using **ffmpeg**, a frame (a *document*) is extracted every second and OCR is performed on it using **tesseract**.
2. Each document is divided into words (*terms*) constituing lines, whose confidence is tesseract's average for each word and whose tf-idf is computed.
3. Each line's tf-idf is multiplied by its confidence.
4. Each line's `(tf-idf * confidence, confidence)` pair is fed into a SVN to determine whether it should be included.
    - Said SVN is trained with a dataset in `weights/*.json`.
5. Included lines are reassembled and outputted in JSON.

## Weights

A JSON file containing a list of training documents, each a list of lines that should or shouldn't be included.

## Examples

```
$ tf_idf.py <lecture> (Output every line and it's tf_idf * confidence*)
$ tf_idf.py <lecture> -w <weights> (Generate a transcript using weights contained in the given JSON file)
$ tf_idf.py <lecture> -w <weights> -r (Generate a transcript, but reuse prior frames and tesseract output)
```
