from preprocessing.clean_text import clean_text

sample_text = "Hello!!!   How are YOU???"

cleaned = clean_text(sample_text)

print("Original:", sample_text)
print("Cleaned:", cleaned)